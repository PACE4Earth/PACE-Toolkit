import os
import json
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta

R_EARTH = 6371000.0
OMEGA = 7.2921e-5

def inspect_nc(path):
    with xr.open_dataset(path) as ds:
        print(ds)

def check_required_fields(path, metric_requirements, aliases):
    ret = {}
    with xr.open_dataset(path, engine='netcdf4') as ds:
        available_fields = list(ds.variables.keys())
    for metric in metric_requirements:
        ret[metric] = None
        for alias in aliases.get(metric, []):
            if alias in available_fields:
                ret[metric] = alias
                break
    return ret

def get_grid(path, lat_range=None, lon_range=None, pressure_levels=None):
    with xr.open_dataset(path, engine='netcdf4') as ds:
        try:
            lats = ds['latitude'].values
        except KeyError:
            lats = ds['lat'].values
        try:
            lons = ds['longitude'].values
        except KeyError:
            lons = ds['lon'].values

        level_dim = 'level' if 'level' in ds.dims else 'pressure' if 'pressure' in ds.dims else None
        if pressure_levels is not None and level_dim:
            levels = ds[level_dim].values[:pressure_levels]
        elif level_dim:
            levels = ds[level_dim].values
        else:
            levels = None

    if lats[0] > lats[-1]:
        lats = lats[::-1]

    if lat_range is not None:
        lat_min, lat_max = sorted(lat_range)
        lats = lats[(lats >= lat_min) & (lats <= lat_max)]
    if lon_range is not None:
        lon_min, lon_max = sorted(lon_range)
        lons = lons[(lons >= lon_min) & (lons <= lon_max)]

    lon_span = lons[-1] - lons[0]
    is_global = lon_span > 359.99

    dlat_deg = np.gradient(lats)
    dphi = np.deg2rad(dlat_deg)

    if is_global:
        dlon_deg = np.gradient(np.concatenate([[lons[-1] - 360], lons, [lons[0] + 360]]))[1:-1]
    else:
        dlon_deg = np.gradient(lons)

    dlambda = np.deg2rad(dlon_deg)
    dy = (R_EARTH * dphi)[:, None] * np.ones((len(lats), len(lons)))
    dx = R_EARTH * np.cos(np.deg2rad(lats))[:, None] * dlambda[None, :]
    f = 2 * OMEGA * np.sin(np.deg2rad(lats))[:, None] * np.ones_like(lons)[None, :]
    f[np.abs(f) < 1e-5] = 1e-5 * np.sign(f[np.abs(f) < 1e-5] + 1e-9)

    return {
        'lon': torch.tensor(lons),
        'lat': torch.tensor(lats),
        'dx': torch.tensor(dx),
        'dy': torch.tensor(dy),
        'f': torch.tensor(f),
        'pressure_levels': torch.tensor(levels) if levels is not None else None
    }

class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, config_path=None, dataset_key='model'):
        aliases_path = Path(__file__).resolve().parent.parent / "configs" / "aliases.json"
        with open(aliases_path, "r") as f:
            self.aliases = json.load(f)

        config_path = Path(config_path) if config_path else Path(__file__).resolve().parent.parent / "configs" / "dataset_config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        def expand_env_vars(obj):
            if isinstance(obj, dict):
                return {k: expand_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_env_vars(i) for i in obj]
            elif isinstance(obj, str):
                return os.path.expandvars(obj)
            else:
                return obj

        self.config = expand_env_vars(self.config)
        dataset_config = self.config["datasets"].get(dataset_key, {})
        self.name = dataset_config.get("name", dataset_key)
        self.is_model_dataset = dataset_key == 'model'
        self.path = dataset_config.get("path", "")

        self.spatial_config = self.config.get("spatial", {})
        self.time_config = self.config.get("time", {})
        self.n_fullfield_samples = self.config.get("visualization", {}).get("n_fullfield_samples", 10)

        self.lat_min, self.lat_max = sorted(self.spatial_config.get("lat_range", [-90, 90]))
        self.lon_min, self.lon_max = sorted(self.spatial_config.get("lon_range", [0, 360]))
        self.pressure_levels = self.spatial_config.get("pressure_levels", None)

        self.start = self.time_config.get("start")
        self.end = self.time_config.get("end")
        self.stride_hours = self.time_config.get("stride_hours", 6)
        self.max_lead = self.time_config.get("lead_times", 1)

        self.start_dt = datetime.strptime(self.start, "%Y%m%d")
        self.end_dt = datetime.strptime(self.end, "%Y%m%d") + timedelta(days=1) - timedelta(minutes=1)
        lead_end_dt = self.end_dt - pd.to_timedelta(self.max_lead * self.stride_hours, unit='h') if self.is_model_dataset else self.end_dt

        print(f'Preparing files for {self.name}...')
        candidate_files = []
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".nc"):
                    path = Path(os.path.join(root, file))
                    candidates = [path.stem, path.parent.name, path.parent.parent.name]
                    for c in candidates:
                        if len(c) >= 8 and c[:8].isdigit():
                            if self.start[:8] <= c[:8] <= self.end[:8]:
                                candidate_files.append(path)


        candidate_files.sort()

        def try_parse_datetime_from_str(s):
            formats = [("%Y%m%d_%H", 3), ("%Y%m%d%H", 3), ("%Y%m%d", 2), ("%Y", 1)]
            for fmt, score in formats:
                try:
                    dt = pd.to_datetime(s, format=fmt)
                    return dt, score
                except Exception:
                    continue
            return None, 0

        self.files = []
        for file in candidate_files:
            path = Path(file)
            # Try to read time variable type first
            try:  
                with xr.open_dataset(path, engine='netcdf4') as ds:
                    time_var = ds['time'].values
                    if np.issubdtype(time_var.dtype, np.datetime64):
                        fallback_base_dt = pd.to_datetime(time_var[0])
                        time_is_absolute = True
                    elif np.issubdtype(time_var.dtype, (np.timedelta64, np.integer)):
                        fallback_base_dt = None
                        time_is_absolute = False
                    else:
                        print(f"Warning: Unsupported time dtype in {file}, skipping.")
                        continue
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue

            # Try to parse base_time from filename only if time is not absolute
            best_dt, best_score = None, 0
            if not time_is_absolute:
                candidates = [path.stem, path.parent.name, path.parent.parent.name]
                for candidate in candidates:
                    dt, score = try_parse_datetime_from_str(candidate)
                    if dt is not None and score > best_score:
                        best_dt, best_score = dt, score

            # Determine base_dt
            base_dt = best_dt if best_dt is not None else fallback_base_dt
            if base_dt is None:
                print(f"Warning: Could not determine base_time for {file}, skipping.")
                continue

            if self.start_dt <= base_dt <= lead_end_dt:
                self.files.append((file, base_dt))


        print(f"Done. Found {len(self.files)} usable files.\n")
        if not self.files:
            raise RuntimeError(f"No input files found for dataset '{dataset_key}' in range {self.start} to {self.end}.")

        print('Static fields setup...', end=' ')
        self.grid = get_grid(
            self.files[0][0],
            lat_range=[self.lat_min, self.lat_max],
            lon_range=[self.lon_min, self.lon_max],
            pressure_levels=self.pressure_levels
        )
        print('Done')

        print('Checking required field for metrics...')
        metrics_path = Path(__file__).resolve().parent.parent / "configs" / "variables_for_metrics.json"
        with open(metrics_path, 'r') as f:
            metrics_requirements = json.load(f)

        self.metrics = {}
        for metric in self.config['metrics']:
            req_for_this_metric = metrics_requirements[metric]
            found = check_required_fields(self.files[0][0], req_for_this_metric, self.aliases)
            if all(f is not None for f in found.values()):
                print(f'{metric:<23} is complete.')
                self.metrics[metric] = found
            else:
                print(f'{metric:<23} is missing field/s for computation.')
        print("\n\n")

        self.canonical_names = list(dict.fromkeys([k for v in self.metrics.values() for k in v]))
        self.requested_names = list(dict.fromkeys([v for d in self.metrics.values() for v in d.values()]))

        self.samples = []
        self.valid_time_map = {}
        for file_path, base_dt in self.files:
            with xr.open_dataset(file_path, engine='netcdf4') as ds:
                time_var = ds['time'].values
                if np.issubdtype(time_var.dtype, np.timedelta64):
                    valid_times = base_dt + pd.to_timedelta(time_var)
                elif np.issubdtype(time_var.dtype, np.datetime64):
                    valid_times = pd.to_datetime(time_var)
                elif np.issubdtype(time_var.dtype, np.integer):
                    valid_times = base_dt + pd.to_timedelta(time_var, unit='h')
                else:
                    raise TypeError(f"Unsupported time dtype: {time_var.dtype}")

                valid_times = valid_times[(valid_times >= self.start_dt) & (valid_times <= self.end_dt)]
                for lead_idx, valid_time in enumerate(valid_times):
                    self.samples.append((file_path, base_dt, lead_idx))
                    self.valid_time_map[(base_dt, lead_idx)] = valid_time

        self.valid_times = sorted(set(self.valid_time_map.values()))
        self.fullfield_sample_flags = [False] * len(self.samples)

    def select_matching_fullfield_samples(self, other_dataset):
        shared_valid_times = sorted(set(self.valid_times) & set(other_dataset.valid_times))
        rng = np.random.default_rng(42)
        chosen_valid_times = rng.choice(
            shared_valid_times,
            size=min(self.n_fullfield_samples, len(shared_valid_times)),
            replace=False
        )
        chosen_valid_times_set = set(chosen_valid_times)
        other_dataset.fullfield_sample_flags = [
            other_dataset.valid_time_map[(base_dt, lead_idx)] in chosen_valid_times_set
            for (_, base_dt, lead_idx) in other_dataset.samples
        ]
        selected_valid_times = set()
        flags = []
        for (file_path, base_dt, lead_idx) in self.samples:
            vt = self.valid_time_map[(base_dt, lead_idx)]
            if vt in chosen_valid_times_set:
                if vt in selected_valid_times:
                    flags.append(False)
                else:
                    flags.append(True)
                    selected_valid_times.add(vt)
            else:
                flags.append(False)
            if len(selected_valid_times) >= len(chosen_valid_times_set):
                flags.extend([False] * (len(self.samples) - len(flags)))
                break
        self.fullfield_sample_flags = flags

        print(self.__len__())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, base_dt, lead_idx = self.samples[idx]
        with xr.open_dataset(file_path, engine='netcdf4') as ds:
            lead_time_val = ds['time'].isel(time=lead_idx).values
            if np.issubdtype(type(lead_time_val), np.timedelta64):
                lead_time = pd.to_timedelta(lead_time_val).to_pytimedelta()
            elif np.issubdtype(type(lead_time_val), np.datetime64):
                lead_time = pd.to_datetime(lead_time_val) - base_dt
            elif np.issubdtype(type(lead_time_val), np.integer):
                lead_time = timedelta(hours=int(lead_time_val))
            else:
                raise TypeError(f"Unsupported lead_time dtype: {type(lead_time_val)}")

            ds = ds.isel(time=lead_idx)

            if 'latitude' in ds:
                ds = ds.sel(latitude=slice(self.lat_max, self.lat_min))
            elif 'lat' in ds:
                ds = ds.sel(lat=slice(self.lat_min, self.lat_max))

            if 'longitude' in ds:
                ds = ds.sel(longitude=slice(self.lon_min, self.lon_max))
            elif 'lon' in ds:
                ds = ds.sel(lon=slice(self.lon_min, self.lon_max))

            level_dim = 'level' if 'level' in ds.dims else 'pressure' if 'pressure' in ds.dims else None
            if self.pressure_levels is not None and level_dim:
                ds = ds.isel({level_dim: slice(0, self.pressure_levels)})

            fields = {}
            for rq, cn in zip(self.requested_names, self.canonical_names):
                tau = torch.tensor(ds[rq].values)
                if tau.ndim == 3:
                    tau = tau.unsqueeze(0)
                fields[cn] = tau

            fields['base_time'] = base_dt
            fields['lead_time'] = lead_time

        return fields

def main():
    config_path = "/p/project/hclimrep/vas1/PACE-Toolkit/pace/configs/dataset_config.json"
    model_dataset = UnifiedDataset(config_path, dataset_key="model")
    reference_dataset = UnifiedDataset(config_path, dataset_key="reference") if "reference" in model_dataset.config.get("datasets", {}) else None

    if reference_dataset:
        model_dataset.select_matching_fullfield_samples(reference_dataset)

    print("\nModel valid times:", model_dataset.valid_times)
    print(f"\n{model_dataset.n_fullfield_samples} randomly selected fullfield samples:")
    for i, (file_path, base_dt, lead_idx) in enumerate(model_dataset.samples):
        if model_dataset.fullfield_sample_flags[i]:
            valid_time = model_dataset.valid_time_map[(base_dt, lead_idx)]
            print(f"Base: {base_dt}, LeadIdx: {lead_idx}, Valid: {valid_time}, File: {file_path.name}")
            sample = model_dataset[i]
            print("  base_time:", sample['base_time'])
            print("  lead_time:", sample['lead_time'])

    if reference_dataset:
        print("\nReference valid times:", reference_dataset.valid_times)
        print(f"\n{reference_dataset.n_fullfield_samples} matching fullfield samples:")
        for i, (file_path, base_dt, lead_idx) in enumerate(reference_dataset.samples):
            if reference_dataset.fullfield_sample_flags[i]:
                valid_time = reference_dataset.valid_time_map[(base_dt, lead_idx)]
                print(f"Base: {base_dt}, LeadIdx: {lead_idx}, Valid: {valid_time}, File: {file_path.name}")
                sample = reference_dataset[i]
                print("  base_time:", sample['base_time'])
                print("  lead_time:", sample['lead_time'])

if __name__ == "__main__":
    main()

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

        if pressure_levels == "all":
            levels = ds[level_dim].values if level_dim else None
        elif isinstance(pressure_levels, list) and level_dim: # Select levels by exact matching pressure values (e.g., [500, 850])
            available_levels = ds[level_dim].values
            indices = [i for i, lvl in enumerate(available_levels) if lvl in pressure_levels]
            levels = available_levels[indices]
        elif pressure_levels is not None and level_dim:  # Assume integer count of levels
            levels = ds[level_dim].values[:pressure_levels]
        elif level_dim:
            levels = ds[level_dim].values
        else:
            levels = None

    if lats[0] > lats[-1]:
        lats = lats[::-1]

    if lat_range is not None and lat_range != "all":
        lat_min, lat_max = sorted(lat_range)
        lats = lats[(lats >= lat_min) & (lats <= lat_max)]

    if lon_range is not None and lon_range != "all":
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

def parse_base_lead_times_from_file(path):
    """
    Returns (base_time: datetime, lead_times: list of timedelta) for the given file path,
    or (None, None) if base_time can't be determined.
    """
    try:
        with xr.open_dataset(path, engine='netcdf4') as ds:
            # Support explicit base_time and lead_time variables if present
            if 'base_time' in ds.variables and 'lead_time' in ds.variables:
                base_time_val = ds['base_time'].values
                lead_time_vals = ds['lead_time'].values
                
                if np.issubdtype(base_time_val.dtype, np.datetime64):
                    base_time = pd.to_datetime(base_time_val).to_pydatetime()
                elif isinstance(base_time_val, (str, bytes)):
                    base_time = pd.to_datetime(base_time_val).to_pydatetime()
                else:
                    base_time = None
                
                # Convert lead_times to timedelta objects
                if np.issubdtype(lead_time_vals.dtype, np.timedelta64):
                    lead_times = [pd.to_timedelta(lt).to_pytimedelta() for lt in lead_time_vals]
                elif np.issubdtype(lead_time_vals.dtype, np.integer):
                    # Assuming lead times in hours
                    lead_times = [timedelta(hours=int(lt)) for lt in lead_time_vals]
                else:
                    lead_times = None

                if base_time is not None and lead_times is not None:
                    return base_time, lead_times

            # Fallback: use 'time' variable
            time_var = ds['time'].values
            if np.issubdtype(time_var.dtype, np.datetime64):
                # time var is absolute datetime, base time = first time, lead times = time - base_time
                base_time = pd.to_datetime(time_var[0]).to_pydatetime()
                lead_times = [pd.to_datetime(t).to_pydatetime() - base_time for t in time_var]
                return base_time, lead_times
            
            elif np.issubdtype(time_var.dtype, (np.timedelta64, np.integer)):
                # time var is relative lead times, parse base_time from filename and dirs
                candidates = [path.stem, path.parent.name, path.parent.parent.name]
                best_dt, best_score = None, 0
                for candidate in candidates:
                    dt, score = try_parse_datetime_from_str(candidate)
                    if dt is not None and score > best_score:
                        best_dt, best_score = dt, score
                base_time = best_dt
                if base_time is None:
                    return None, None
                # Convert lead times to timedelta list
                if np.issubdtype(time_var.dtype, np.timedelta64):
                    lead_times = [pd.to_timedelta(t).to_pytimedelta() for t in time_var]
                else:
                    lead_times = [timedelta(hours=int(t)) for t in time_var]
                return base_time, lead_times
            
            else:
                # Unsupported time dtype
                return None, None

    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None, None

def try_parse_datetime_from_str(s):
    formats = [("%Y%m%d_%H", 3), ("%Y%m%d%H", 3), ("%Y%m%d", 2), ("%Y", 1)]
    for fmt, score in formats:
        try:
            dt = pd.to_datetime(s, format=fmt)
            return dt, score
        except Exception:
            continue
    return None, 0


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, config_path=None, dataset_key='model', shared_valid_times=None):
        aliases_path = Path(__file__).resolve().parent.parent / "configs" / "aliases.json"
        with open(aliases_path, "r") as f:
            self.aliases = json.load(f)

        config_path = Path(config_path) if config_path else Path(__file__).resolve().parent.parent / "configs" / "config.json"
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
        lat_range = self.spatial_config.get("lat_range", "all")
        lon_range = self.spatial_config.get("lon_range", "all")
        self.lat_min, self.lat_max = (None, None) if lat_range == "all" else sorted(lat_range)
        self.lon_min, self.lon_max = (None, None) if lon_range == "all" else sorted(lon_range)
        self.pressure_levels = self.spatial_config.get("pressure_levels", None)

        self.time_config = self.config.get("time", {})
        self.start = self.time_config.get("start")
        self.end = self.time_config.get("end")
        self.stride_hours = self.time_config.get("stride_hours", 6)
        self.sample_percent = self.time_config.get("sample_percent", 1)
        self.max_lead = self.time_config.get("num_lead_times", 1)

        # Custom times config
        self.custom_times_enabled = self.time_config.get("custom_times", {}).get("enabled", False)
        self.custom_times_list = []
        if self.custom_times_enabled:
            custom_list_raw = self.time_config.get("custom_times", {}).get("times", [])
            for ct in custom_list_raw:
                try:
                    self.custom_times_list.append(datetime.strptime(ct, "%Y%m%d_%H"))
                except ValueError:
                    raise ValueError(f"Invalid custom time format: {ct}, expected YYYYMMDD_HH")

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

        self.files = []
        for file in candidate_files:
            base_dt, lead_times = parse_base_lead_times_from_file(file)
            if base_dt is None or lead_times is None:
                print(f"Warning: Could not determine base or lead times for {file}, skipping.")
                continue
            
            lead_end_dt = self.end_dt - pd.to_timedelta(self.max_lead * self.stride_hours, unit='h') if self.is_model_dataset else self.end_dt
            if not (self.start_dt <= base_dt <= lead_end_dt):
                continue

            self.files.append((file, base_dt, lead_times))

        print(f"Done. Found {len(self.files)} usable files.\n")
        if not self.files:
            raise RuntimeError(f"No input files found for dataset '{dataset_key}' in range {self.start} to {self.end}.")

        print('Static fields setup...', end=' ')
        self.grid = get_grid(
            self.files[0][0],
            lat_range=None if self.lat_min is None else [self.lat_min, self.lat_max],
            lon_range=None if self.lon_min is None else [self.lon_min, self.lon_max],
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

        self.canonical_names = list(dict.fromkeys([k for v in self.metrics.values() for k in v]))
        self.requested_names = list(dict.fromkeys([v for d in self.metrics.values() for v in d.values()]))

        self.samples = []
        self.valid_time_map = {}
        for file_path, base_dt, lead_times in self.files:
            valid_times = [base_dt + lt for lt in lead_times]
            valid_times = [vt for vt in valid_times if self.start_dt <= vt <= self.end_dt]

            max_lead_idx = min(self.max_lead, len(valid_times))
            for lead_idx in range(max_lead_idx):
                valid_time = valid_times[lead_idx]
                self.samples.append((file_path, base_dt, lead_idx, lead_times)) # Store lead_times array in the sample
                self.valid_time_map[(base_dt, lead_idx)] = valid_time

        self.valid_times = sorted(set(self.valid_time_map.values()))

        # Select custom or random valid times
        if self.custom_times_enabled:
            chosen_valid_times = set()
            for target_time in self.custom_times_list:
                closest_time = min(self.valid_times, key=lambda t: abs(t - target_time))
                chosen_valid_times.add(closest_time)
        else:
            rng = np.random.default_rng(42)
            if shared_valid_times is not None:
                chosen_valid_times = shared_valid_times
            else:
                num_samples = max(1, round(len(self.valid_times) * (self.sample_percent / 100.0)))
                chosen_valid_times = set(rng.choice(self.valid_times, size=num_samples, replace=False))

        # Filter samples
        self.samples = [
            (fp, bd, li, lts)
            for (fp, bd, li, lts) in self.samples
            if self.valid_time_map[(bd, li)] in chosen_valid_times
        ]

        self.chosen_valid_times = chosen_valid_times 

        print(f"\nSelected {len(self.samples)} samples for {self.name}.")
        print("---------------------------------------------\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, base_dt, lead_idx, lead_times = self.samples[idx]
        lead_time = lead_times[lead_idx]
        with xr.open_dataset(file_path, engine='netcdf4') as ds:
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
            if level_dim and self.pressure_levels is not None:
                if self.pressure_levels == "all":
                    pass  # No slicing, select all levels
                elif isinstance(self.pressure_levels, list):
                    available_levels = ds[level_dim].values
                    indices = [i for i, lvl in enumerate(available_levels) if lvl in self.pressure_levels]
                    ds = ds.isel({level_dim: indices})
                else:
                    ds = ds.isel({level_dim: slice(0, self.pressure_levels)})  # assume integer count slicing

            fields = {}
            for rq, cn in zip(self.requested_names, self.canonical_names):
                tau = torch.tensor(ds[rq].values)
                if tau.ndim == 3:
                    tau = tau.unsqueeze(0)
                fields[cn] = tau

            fields['base_time'] = base_dt
            fields['lead_time'] = lead_time

        return fields
    
    # Construct from a precomputed sample list and minimal metadata
    @classmethod
    def from_sample_list(cls, sample_list, grid, metrics, requested_names, canonical_names, config_path=None, dataset_key='model'):
        obj = cls.__new__(cls)

        # Reassign basic metadata
        obj.samples = sample_list
        obj.grid = grid
        obj.metrics = metrics
        obj.requested_names = requested_names
        obj.canonical_names = canonical_names
        obj.name = dataset_key

        # Minimal needed for __getitem__
        obj.config = {}
        obj.lat_min = float(grid['lat'].min())
        obj.lat_max = float(grid['lat'].max())
        obj.lon_min = float(grid['lon'].min())
        obj.lon_max = float(grid['lon'].max())
        obj.pressure_levels = None if grid['pressure_levels'] is None else (
            int(grid['pressure_levels'].item()) if grid['pressure_levels'].ndim == 0 
            else grid['pressure_levels'].tolist()
        )   

        obj.is_model_dataset = dataset_key == 'model'

        return obj

def main():
    config_path = "/p/project/hclimrep/vas1/PACE-Toolkit/pace/configs/config.json"
    model_dataset = UnifiedDataset(config_path, dataset_key="model")
    reference_dataset = UnifiedDataset(config_path, dataset_key="reference", shared_valid_times=model_dataset.chosen_valid_times) if "reference" in model_dataset.config.get("datasets", {}) else None
    print(f"len model: {model_dataset.__len__()}")
    print(model_dataset.samples)
    # print(model_dataset.grid["pressure_levels"])
    # print(model_dataset.grid["lat"])
    # print(model_dataset.grid["lon"])

    if reference_dataset:
        print(f"len ref: {reference_dataset.__len__()}")
        # print(model_dataset.grid["pressure_levels"])
        # print(model_dataset.grid["lat"])
        # print(model_dataset.grid["lon"])

    print("\nModel valid times:", model_dataset.valid_times)
    for i, (file_path, base_dt, lead_idx, leadtimes) in enumerate(model_dataset.samples):
        valid_time = model_dataset.valid_time_map[(base_dt, lead_idx)]
        print(f"Base: {base_dt}, LeadIdx: {lead_idx}, Valid: {valid_time}, File: {file_path.name}")
        sample = model_dataset[i]
        print("  base_time:", sample['base_time'])
        print("  lead_time:", sample['lead_time'])

    if reference_dataset:
        print("\nReference valid times:", reference_dataset.valid_times)
        for i, (file_path, base_dt, lead_idx, leadtimes) in enumerate(reference_dataset.samples):
            valid_time = reference_dataset.valid_time_map[(base_dt, lead_idx)]
            print(f"Base: {base_dt}, LeadIdx: {lead_idx}, Valid: {valid_time}, File: {file_path.name}")
            sample = reference_dataset[i]
            print("  base_time:", sample['base_time'])
            print("  lead_time:", sample['lead_time'])

if __name__ == "__main__":
    main()

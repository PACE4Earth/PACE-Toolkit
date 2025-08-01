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
            config = json.load(f)

        def expand_env_vars(obj):
            if isinstance(obj, dict):
                return {k: expand_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_env_vars(i) for i in obj]
            elif isinstance(obj, str):
                return os.path.expandvars(obj)
            else:
                return obj

        config = expand_env_vars(config)
        dataset_config = config["datasets"].get(dataset_key, {})
        self.name = dataset_config.get("name", dataset_key)
        self.is_model_dataset = dataset_key == 'model'
        self.path = dataset_config.get("path", "")

        self.spatial_config = config.get("spatial", {})
        self.time_config = config.get("time", {})

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
                        if c[:8].isdigit() and self.start[:8] <= c[:8] <= self.end[:8]:
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
            candidates = [path.stem, path.parent.name, path.parent.parent.name]
            best_dt, best_score = None, 0
            for candidate in candidates:
                dt, score = try_parse_datetime_from_str(candidate)
                if dt is not None and score > best_score:
                    best_dt, best_score = dt, score

            if best_dt and self.start_dt <= best_dt <= lead_end_dt:
                self.files.append((file, best_dt))

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
        for metric in config['metrics']:
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
                for lead_idx in range(len(valid_times)):
                    self.samples.append((file_path, base_dt, lead_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, base_dt, lead_idx = self.samples[idx]
        with xr.open_dataset(file_path, engine='netcdf4') as ds:
            time_vals = ds['time'].values
            if np.issubdtype(time_vals.dtype, np.timedelta64):
                lead_time_dt = base_dt + pd.to_timedelta(time_vals[lead_idx])
            elif np.issubdtype(time_vals.dtype, np.datetime64):
                lead_time_dt = pd.to_datetime(time_vals[lead_idx])
            elif np.issubdtype(time_vals.dtype, np.integer):
                lead_time_dt = base_dt + pd.to_timedelta(time_vals[lead_idx], unit='h')
            else:
                raise TypeError(f"Unsupported time dtype: {time_vals.dtype}")

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

            base_time_ts = torch.tensor(base_dt.to_datetime64().astype('datetime64[s]').astype(np.int64))
            lead_time_ts = torch.tensor(lead_time_dt.to_datetime64().astype('datetime64[s]').astype(np.int64))
            fields['base_time'] = base_time_ts
            fields['lead_time'] = lead_time_ts

        return fields

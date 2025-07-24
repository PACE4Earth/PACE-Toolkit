import os
import json
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path
from datetime import datetime

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


def get_grid(path, lat_range=None, lon_range=None):
    with xr.open_dataset(path, engine='netcdf4') as ds:
        try:
            lats = ds['latitude'].values
        except KeyError:
            lats = ds['lat'].values
        try:
            lons = ds['longitude'].values
        except KeyError:
            lons = ds['lon'].values

    if lat_range is not None:
        lat_min, lat_max = sorted(lat_range)
        lats = lats[(lats >= lat_min) & (lats <= lat_max)]
    if lon_range is not None:
        lon_min, lon_max = sorted(lon_range)
        lons = lons[(lons >= lon_min) & (lons <= lon_max)]

    dlat_deg = np.gradient(lats)
    dlon_deg = np.gradient(np.concatenate([[lons[-1]-360], lons, [lons[0]+360]]))[1:-1]
    dphi = np.deg2rad(dlat_deg)
    dlambda = np.deg2rad(dlon_deg)
    
    dy = R_EARTH * dphi
    dy = dy[:, None] * np.ones_like(lons)[None, :]
    dx = R_EARTH * np.cos(np.deg2rad(lats))[:, None] * dlambda[None, :]
    
    f = 2 * OMEGA * np.sin(np.deg2rad(lats))[:, None] * np.ones_like(lons)[None, :]
    f[np.abs(f) < 1e-5] = 1e-5 * np.sign(f[np.abs(f) < 1e-5] + 1e-9)

    return {
        'lon': torch.tensor(lons),
        'lat': torch.tensor(lats),
        'dx': torch.tensor(dx),
        'dy': torch.tensor(dy),
        'f': torch.tensor(f),
    }


class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, config_path=None, dataset_key='model'):
        # Load variable aliases
        aliases_path = Path(__file__).resolve().parent.parent / "configs" / "aliases.json"
        with open(aliases_path, "r") as f:
            self.aliases = json.load(f)

        # Load config
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent / "configs" / "graphcast_extended.json"
        else:
            config_path = Path(config_path)

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
        self.path = dataset_config.get("path", "")

        self.spatial_config = config.get("spatial", {})
        self.time_config = config.get("time", {})

        self.lat_min, self.lat_max = sorted(self.spatial_config.get("lat_range", [-90, 90]))
        self.lon_min, self.lon_max = sorted(self.spatial_config.get("lon_range", [0, 360]))
        self.pressure_levels = self.spatial_config.get("pressure_levels", None)

        self.start = self.time_config.get("start")
        self.end = self.time_config.get("end")
        self.stride_hours = self.time_config.get("stride_hours", 6)  # Scalar
        self.max_lead = self.time_config.get("lead_times", 1)  # Scalar, e.g. 40

        self.start_dt = datetime.strptime(self.start, "%Y%m%d")
        self.end_dt = datetime.strptime(self.end, "%Y%m%d")

        # Determine latest possible base time to ensure full forecast fits in range
        lead_end_dt = self.end_dt - pd.to_timedelta(self.max_lead * self.stride_hours, unit='h')

        print('Preparing files...', end=' ')
        candidate_files = []
        for root, dirs, files in os.walk(self.path):
            for f in files:
                if not f.endswith(".nc"):
                    continue
                fname_dt = f[:8]
                dirname_dt = os.path.basename(root)[:8]
                if (fname_dt.isdigit() and self.start <= fname_dt <= self.end) or \
                   (dirname_dt.isdigit() and self.start <= dirname_dt <= self.end):
                    candidate_files.append(os.path.join(root, f))
        candidate_files.sort()

        # Filter files based on whether their base_dt + full leadtime fits within time range
        self.files = []
        for file in candidate_files:
            try:
                parent_name = Path(file).parent.name
                try:
                    base_dt = pd.to_datetime(parent_name, format="%Y%m%d_%H")
                except Exception:
                    base_dt = pd.to_datetime(parent_name, format="%Y%m%d")

                if not (self.start_dt <= base_dt <= lead_end_dt):
                    continue

                self.files.append((file, base_dt))

            except Exception as e:
                print(f"Error reading {file}: {e}")

        print(f"Done. Found {len(self.files)} usable files.")

        if not self.files:
            raise RuntimeError(f"No input files found for dataset '{dataset_key}' in range {self.start} to {self.end}.")

        print('Static fields setup...', end=' ')
        self.grid = get_grid(
            self.files[0][0],
            lat_range=[self.lat_min, self.lat_max],
            lon_range=[self.lon_min, self.lon_max]
        )
        print('Done')

        print('Checking required field for metrics...')
        metrics_path = Path(__file__).resolve().parent.parent / "configs" / "variables_for_metrics.json"
        with open(metrics_path, 'r') as f:
            metrics_requirements = json.load(f)

        self.metrics = {}
        for metric in config['metrics']:
            req_for_this_metric = metrics_requirements[metric]
            found_for_this_metric = check_required_fields(self.files[0][0], req_for_this_metric, self.aliases)
            if any(f is None for f in found_for_this_metric.values()):
                print(f'{metric:<23} is missing field/s for computation.')
            else:
                print(f'{metric:<23} is complete.')
                self.metrics[metric] = found_for_this_metric

        names_in_files = []
        canonical_names = []
        for v in self.metrics.values():
            for canonical, true in v.items():
                canonical_names.append(canonical)
                names_in_files.append(true)
        self.canonical_names = list(dict.fromkeys(canonical_names))
        self.requested_names = list(dict.fromkeys(names_in_files))

        print('\nFields to be loaded:')
        for cn, rn in zip(self.canonical_names, self.requested_names):
            print(f'{cn:<20} <- {rn}')

        # Create sample list: each sample corresponds to one lead time from one file
        self.samples = []
        for file_path, base_dt in self.files:
            for lead_idx in range(self.max_lead):
                self.samples.append((file_path, base_dt, lead_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, base_dt, lead_idx = self.samples[idx]

        with xr.open_dataset(file_path, engine='netcdf4') as ds:
            # Extract the time coordinate array
            time_vals = ds['time'].values

            # Interpret the time values properly
            if np.issubdtype(time_vals.dtype, np.timedelta64):
                valid_times = base_dt + pd.to_timedelta(time_vals)
            elif np.issubdtype(time_vals.dtype, np.datetime64):
                valid_times = pd.to_datetime(time_vals)
            elif np.issubdtype(time_vals.dtype, np.integer):
                valid_times = base_dt + pd.to_timedelta(time_vals, unit='h')
            else:
                raise TypeError(f"Unsupported time dtype: {time_vals.dtype}")

            # Select only the requested lead time slice
            ds = ds.isel(time=lead_idx)

            # Select spatial slices
            if 'latitude' in ds:
                ds = ds.sel(latitude=slice(self.lat_max, self.lat_min))
            elif 'lat' in ds:
                ds = ds.sel(lat=slice(self.lat_min, self.lat_max))

            if 'longitude' in ds:
                ds = ds.sel(longitude=slice(self.lon_min, self.lon_max))
            elif 'lon' in ds:
                ds = ds.sel(lon=slice(self.lon_min, self.lon_max))

            fields = {}
            for rq, cn in zip(self.requested_names, self.canonical_names):
                tau = torch.tensor(ds[rq].values)
                # If 3D (e.g. [P, H, W]), add channel dimension for consistency
                if tau.ndim == 3:
                    tau = tau.unsqueeze(0)  # [C=1, P, H, W]
                fields[cn] = tau

            # base_time as integer timestamp (seconds since epoch)
            base_time_ts = torch.tensor(base_dt.to_datetime64().astype('datetime64[s]').astype(np.int64))

            # lead_time timestamp = base_time + lead_idx * stride_hours (seconds since epoch)
            lead_time_dt = base_dt + pd.Timedelta(hours=lead_idx * self.stride_hours)
            lead_time_ts = torch.tensor(lead_time_dt.to_datetime64().astype('datetime64[s]').astype(np.int64))

            fields['base_time'] = base_time_ts
            fields['lead_time'] = lead_time_ts

            # Limit pressure levels if requested
            if self.pressure_levels is not None:
                level_dim = 'level' if 'level' in ds.dims else 'pressure' if 'pressure' in ds.dims else None
                if level_dim:
                    ds = ds.isel({level_dim: slice(0, self.pressure_levels)})
                    fields['pressure_levels'] = torch.tensor(ds[level_dim].values)

        return fields

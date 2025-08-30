"""
Dataset Loader

This module provides utilities and a PyTorch Dataset class (`UnifiedDataset`)
to handle loading of model and reference datasets (e.g., GraphCast, ERA5, CorrDiff).
It supports:
    - Loading files from directories
    - Spatial/temporal subsetting
    - Pressure level selection
    - Aliasing variable names across datasets
    - Sampling valid times with random or custom selection
    - Preparing inputs for metric computation
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import time

from .parsers.netcdf import parse_directory
from .parsers.netcdf_groups import parse_directory_groups

# --- Physical constants ---
R_EARTH = 6371000.0  # Radius of Earth (m)
OMEGA = 7.2921e-5    # Earth's angular velocity (rad/s)


# ----------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------

def check_required_fields(path, opener_kwargs, metric_requirements, aliases):
    """
    Check that required fields (variables) for metric computation exist in the dataset.

    Parameters
    ----------
    path : str or Path
        Path to the NetCDF file.
    opener_kwargs : dict
        Arguments passed to `xr.open_dataset` (e.g., engine, group).
    metric_requirements : dict
        Mapping of metric → list of required fields.
    aliases : dict
        Mapping of canonical field names → list of possible aliases.

    Returns
    -------
    dict
        Mapping of metric → resolved field names (or None if missing).
    """
    ret = {}
    with xr.open_dataset(path, **opener_kwargs) as ds:
        available_fields = list(ds.variables.keys())

    for metric in metric_requirements:
        ret[metric] = None
        for alias in aliases.get(metric, []):
            if alias in available_fields:
                ret[metric] = alias
                break
    return ret

def get_grid(path, opener_kwargs, lat_range=None, lon_range=None, pressure_levels=None):
    """
    Extract grid information (lat, lon, dx, dy, Coriolis parameter, pressure levels).

    Parameters
    ----------
    path : str or Path
        Path to a NetCDF file (used to infer grid metadata).
    opener_kwargs : dict
        Arguments for `xr.open_dataset`.
    lat_range, lon_range : list or "all", optional
        Subset of latitude/longitude values.
    pressure_levels : list, int, str, or None
        - "all" → all available levels
        - list → exact matching values (e.g. [500, 850])
        - int → number of levels from the top
        - None → no selection

    Returns
    -------
    dict
        Dictionary with keys: lon, lat, dx, dy, f, pressure_levels (torch tensors).
    """
    # Load dataset (only one opener_kwargs is expected)
    with xr.open_dataset(path, **dict([next(iter(opener_kwargs.items()))])) as ds:
        # Latitude and longitude
        try:
            lats = ds['latitude'].values
        except KeyError:
            lats = ds['lat'].values
        try:
            lons = ds['longitude'].values
        except KeyError:
            lons = ds['lon'].values

        # Identify pressure/level dimension
        level_dim = 'level' if 'level' in ds.dims else 'pressure' if 'pressure' in ds.dims else None

        # Select pressure levels
        if pressure_levels == "all":
            levels = ds[level_dim].values if level_dim else None
        elif isinstance(pressure_levels, list) and level_dim: 
            available_levels = ds[level_dim].values
            indices = [i for i, lvl in enumerate(available_levels) if lvl in pressure_levels]
            levels = available_levels[indices]
        elif pressure_levels is not None and level_dim:  
            levels = ds[level_dim].values[:pressure_levels]  # first N levels
        elif level_dim:
            levels = ds[level_dim].values
        else:
            levels = None

    # Handle 2D lat/lon grids (convert to 1D)        
    if lats.ndim == 2:
        lats = lats[:, 0]
        lons = lons[0, :]

    # Apply spatial subsetting
    if lat_range is not None and lat_range != "all":
        lat_min, lat_max = sorted(lat_range)
        lats = lats[(lats >= lat_min) & (lats <= lat_max)]

    if lon_range is not None and lon_range != "all":
        lon_min, lon_max = sorted(lon_range)
        lons = lons[(lons >= lon_min) & (lons <= lon_max)]

    # Global grid detection
    lon_span = lons[-1] - lons[0]
    is_global = lon_span > 359.99

    # Compute grid spacing
    dlat_deg = np.gradient(lats)
    dphi = np.deg2rad(dlat_deg)

    if is_global:
        # Wrap-around for global longitude
        dlon_deg = np.gradient(np.concatenate([[lons[-1] - 360], lons, [lons[0] + 360]]))[1:-1]
    else:
        dlon_deg = np.gradient(lons)

    dlambda = np.deg2rad(dlon_deg)

    # Physical grid spacings
    dy = (R_EARTH * dphi)[:, None] * np.ones((len(lats), len(lons)))
    dx = R_EARTH * np.cos(np.deg2rad(lats))[:, None] * dlambda[None, :]

    # Coriolis parameter (avoid exactly zero near equator)
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


# ----------------------------------------------------------------------
# UnifiedDataset
# ----------------------------------------------------------------------

class UnifiedDataset(torch.utils.data.Dataset):
    """
    Unified PyTorch Dataset for loading model and reference datasets.

    Features:
    ---------
    - Reads dataset configuration from JSON.
    - Supports environment variable overrides for paths.
    - Parses NetCDF files via custom parsers (e.g., `parse_directory`, `parse_directory_groups`).
    - Applies spatial/temporal subsetting and pressure level selection.
    - Random or custom sampling of valid times.
    - Ensures required variables for metrics are available.

    Each sample returned by __getitem__ contains:
        - Requested fields as torch tensors
        - Metadata: base_time, lead_time, idx
    """

    def __init__(self, config_path=None, dataset_key='model', shared_valid_times=None):
        """
        Parameters
        ----------
        config_path : str or Path, optional
            Path to config JSON. Defaults to config_template.json.
        dataset_key : str
            "model" or "reference" dataset type.
        shared_valid_times : set of datetime, optional
            Shared valid times (used if reference dataset available).
        """

        # --- Load aliases for variable names ---        
        aliases_path = Path(__file__).resolve().parent.parent / "configs" / "aliases.json"
        with open(aliases_path, "r") as f:
            self.aliases = json.load(f)

        # --- Load config ---
        config_path = Path(config_path) if config_path else Path(__file__).resolve().parent.parent / "configs" / "config_template.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Expand environment variables in config
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
        
        # Dataset path (env var overrides config)
        try:
            self.path = os.environ[f"{self.name.upper()}_PATH"]
        except:
            self.path = dataset_config.get("path", "")
        
        # --- Spatial config ---
        self.spatial_config = self.config.get("spatial", {})
        lat_range = self.spatial_config.get("lat_range", "all")
        lon_range = self.spatial_config.get("lon_range", "all")
        self.lat_min, self.lat_max = (None, None) if lat_range == "all" else sorted(lat_range)
        self.lon_min, self.lon_max = (None, None) if lon_range == "all" else sorted(lon_range)
        self.pressure_levels = self.spatial_config.get("pressure_levels", None)

        # --- Time config ---
        self.time_config = self.config.get("time", {})
        self.start = self.time_config.get("start")
        self.end = self.time_config.get("end")
        self.stride_hours = self.time_config.get("stride_hours", 6)
        self.sample_percent = self.time_config.get("sample_percent", 1)
        self.max_lead = self.time_config.get("num_lead_times", 1)

        # Handle custom times (if enabled)
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

        # --- File discovery ---
        print(f'Preparing files for {self.name}...')
        # Selecting correct parser
        if ("graphcast" in self.name) or ("era" in self.name):
            candidate_files = parse_directory(self.path, self.start, self.end)
        elif ("corrdiff" in self.name) or ("rea" in self.name):
            if "corrdiff" in self.name:
                group = "prediction"
            elif "rea" in self.name:
                group = "truth"
            candidate_files = parse_directory_groups(self.path, self.start, self.end, group=group)

        # File filtering based on valid range
        self.files = []
        for file_path, base_time, lead_times, opener_kwargs in candidate_files:
            """ --- Use this option if each base_time should have equal number of leadtimes --- """
            # lead_end_dt = self.end_dt - pd.to_timedelta(self.max_lead * self.stride_hours, unit='h') if self.is_model_dataset else self.end_dt
            # within_lead_range =  self.start_dt <= base_time <= lead_end_dt
            # if within_lead_range:
            #   self.files.append((file_path, base_time, lead_times, opener_kwargs))
            
            """ --- Use this option to include all available samples within time range --- """
            within_valid_range = False
            for lead_time in lead_times:
                valid_time = base_time + lead_time
                within_valid_range = self.start_dt <= valid_time <= self.end_dt
                if within_valid_range:
                    self.files.append((file_path, base_time, lead_times, opener_kwargs))
                    break

        print(f"Done. Found {len(self.files)} usable files ({self.start_dt} -- {self.end_dt}).\n")
        if not self.files:
            raise RuntimeError(f"No input files found for dataset '{dataset_key}' in range {self.start} to {self.end}.")

        # --- Grid setup ---
        print('Grid setup...', end=' ')
        self.grid = get_grid(
            self.files[0][0], self.files[0][3],
            lat_range=None if self.lat_min is None else [self.lat_min, self.lat_max],
            lon_range=None if self.lon_min is None else [self.lon_min, self.lon_max],
            pressure_levels=self.pressure_levels
        )
        print('Done.')

        # Check required fields for metrics
        print('Checking required fields for metrics...')
        metrics_path = Path(__file__).resolve().parent.parent / "configs" / "variables_for_metrics.json"
        with open(metrics_path, 'r') as f:
            metrics_requirements = json.load(f)

        self.metrics = {}
        for metric in self.config['metrics']:
            req_for_this_metric = metrics_requirements[metric]
            found = check_required_fields(self.files[0][0], self.files[0][3], req_for_this_metric, self.aliases)
            if all(f is not None for f in found.values()):
                print(f'{metric:<23} is complete.')
                self.metrics[metric] = found
            else:
                print(f'{metric:<23} is missing field/s for computation.')

        self.canonical_names = list(dict.fromkeys([k for v in self.metrics.values() for k in v]))
        self.requested_names = list(dict.fromkeys([v for d in self.metrics.values() for v in d.values()]))

        # --- Sample creation ---
        self.samples = []
        self.valid_times_for_samples = []
        for file_path, base_time, lead_times, opener_kwargs in self.files:
            lt_vt_pairs = [(i, lt, base_time + lt) for i, lt in enumerate(lead_times)]
            
            # Crop to first N original indices of leadtimes if model dataset
            if self.is_model_dataset:
                lt_vt_pairs = [(i, lt, vt) for i, lt, vt in lt_vt_pairs if i < self.max_lead]

            # Time filter
            lt_vt_pairs = [(i, lt, vt) for i, lt, vt in lt_vt_pairs if self.start_dt <= vt <= self.end_dt]

            for i, lead_time, valid_time in lt_vt_pairs:
                self.samples.append((file_path, base_time, i, lead_time, opener_kwargs))
                self.valid_times_for_samples.append(valid_time)

        self.valid_times = sorted(set(self.valid_times_for_samples))

        # Select custom or random valid times
        if self.custom_times_enabled:
            chosen_valid_times = set()
            for target_time in self.custom_times_list:
                if target_time in self.valid_times:
                    chosen_valid_times.add(target_time)
                else:
                    raise ValueError(f"Target time {target_time} not found in valid_times. Please check time range and num_leadtimes in config.")

        else:
            rng = np.random.default_rng(42)
            if shared_valid_times is not None:
                chosen_valid_times = shared_valid_times
            else:
                num_samples = max(1, round(len(self.valid_times) * (self.sample_percent / 100.0)))
                chosen_valid_times = set(rng.choice(self.valid_times, size=num_samples, replace=False))

        # Filter samples to chosen times
        self.samples = [s for s, vt in zip(self.samples, self.valid_times_for_samples) if vt in chosen_valid_times]
        self.valid_times_for_samples = [vt for vt in self.valid_times_for_samples if vt in chosen_valid_times]
        self.chosen_valid_times = chosen_valid_times 

        print(f"\nSelected {len(self.samples)} samples for {self.name}.")
        print("---------------------------------------------\n")

        # --- Build index map (base_time, lead_time) → global idx ---
        self.index_map = {(base_time, lead_time): i for i, (_, base_time, _, lead_time, _) in enumerate(self.samples)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load one sample as a dict of tensors and metadata.
        """
        file_path, base_time, lead_idx, lead_time, opener_kwargs = self.samples[idx]
        global_idx = self.index_map[(base_time, lead_time)]
        
        with xr.open_dataset(file_path, **opener_kwargs) as ds: 
            if self.name == "corrdiff" or "rea" in self.name:
                # CorrDiff time is integer steps of 6h from 2013-01-01 00:00
                corr_start = datetime(2013, 1, 1, 0, 0)
                step_hours = 6
                delta = (base_time - corr_start)
                step_index = int(delta.total_seconds() // (step_hours * 3600))
                ds = ds.isel(time=step_index)
            else:
                # Default: select by lead index
                ds = ds.isel(time=lead_idx)

            # Spatial selection
            if 'latitude' in ds:
                ds = ds.sel(latitude=slice(self.lat_min, self.lat_max))
            elif 'lat' in ds:
                ds = ds.sel(lat=slice(self.lat_min, self.lat_max))

            if 'longitude' in ds:
                ds = ds.sel(longitude=slice(self.lon_min, self.lon_max))
            elif 'lon' in ds:
                ds = ds.sel(lon=slice(self.lon_min, self.lon_max))

            # Pressure level selection
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

            # Build fields dict
            fields = {}
            for rq, cn in zip(self.requested_names, self.canonical_names):
                tau = torch.tensor(ds[rq].values)
                if tau.ndim == 3:
                    tau = tau.unsqueeze(0)  # Add singleton channel (batch) dimension
                fields[cn] = tau.to(os.getenv('DEVICE'))

            # Metadata
            fields['base_time'] = base_time
            fields['lead_time'] = lead_time
            fields['idx'] = global_idx*torch.ones(1, device=os.getenv('DEVICE')) # just because you are

        return fields
    
    # ------------------------------------------------------------------
    # Alternate constructor (used for other ranks except rank 0)
    # ------------------------------------------------------------------
    @classmethod
    def from_sample_list(
            cls, 
            sample_list, 
            grid, metrics, 
            requested_names, 
            canonical_names, 
            config_path=None, 
            dataset_key='model',
            index_map=None,
        ):
        """
        Construct dataset directly from a prepared sample list and metadata.
        Useful for distributed evaluation (avoids re-parsing).

        Parameters
        ----------
        sample_list : list
            List of samples (tuples).
        grid : dict
            Grid metadata (output of get_grid).
        metrics : dict
            Metrics with required field mappings.
        requested_names, canonical_names : list
            Variable names.
        config_path : str or Path, optional
            Config file
        """
        
        obj = cls.__new__(cls)

        # Reassign basic metadata
        obj.samples = sample_list
        obj.grid = grid
        obj.metrics = metrics
        obj.requested_names = requested_names
        obj.canonical_names = canonical_names
        obj.name = dataset_key
        obj.index_map = index_map

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

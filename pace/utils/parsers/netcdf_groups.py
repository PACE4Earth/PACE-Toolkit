import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from datetime import timedelta

# GROUP_NAMES = ["truth", "prediction", "input"]
# GROUP_NAMES = ["prediction"]

def parse_file(file_path: Path, group: str):
    """
    Parses a single NetCDF file with groups.
    Returns a list of tuples:
        (file_path, base_time, lead_times, opener_kwargs)
    """
    results = []

    # Extract lead time from filename (only once per file)
    match = re.search(r'(\d+)h', str(file_path))
    lead_time = timedelta(hours=int(match.group(1))) if match else timedelta(hours=0)
    
    # Open root dataset to get the global 'time' variable (valid_time)
    try:
        with xr.open_dataset(file_path, engine="netcdf4") as ds_root:
            base_times = pd.to_datetime(ds_root["time"].values).to_pydatetime()
    except Exception as e:
        print(f"Failed to read root dataset {file_path}: {e}")
        return results

    # For each group, append tuples
    # for group in GROUP_NAMES:
    opener_kwargs = {"engine": "netcdf4", "group": group}
    try:
        with xr.open_dataset(file_path, **opener_kwargs) as ds_group:
            for bt in base_times:
                results.append((file_path, bt, [lead_time], opener_kwargs))
    except Exception as e:
        print(e)
        # continue
    
    return results

def parse_directory_groups(base_dir, start=None, end=None, group="prediction"):
    """
    Walks a directory tree and returns a list of all NetCDF files
    successfully parsed. Each entry is (file_path, base_time, lead_times, opener_kwargs).
    """
    results = []
    for file in os.listdir(base_dir):
        if not file.endswith(".nc"):
            continue

        path = Path(base_dir) / file
        parsed = parse_file(path, group=group)
        if parsed is not None:
            results.extend(parsed)
                
    return results

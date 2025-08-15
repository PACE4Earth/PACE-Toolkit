import os
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
from datetime import timedelta

def try_parse_datetime_from_str(s):
    formats = [("%Y%m%d_%H", 3), ("%Y%m%d%H", 3), ("%Y%m%d", 2), ("%Y", 1)]
    for fmt, score in formats:
        try:
            dt = pd.to_datetime(s, format=fmt)
            return dt, score
        except Exception:
            continue
    return None, 0

def parse_file(file_path: Path):
    """
    Returns (file_path, base_time, lead_times, opener_kwargs) for a NetCDF file.
    Assumes one base_time per file.
    Returns None if parsing fails.
    """
    opener_kwargs = {"engine": "netcdf4"} 
    try:
        with xr.open_dataset(file_path, **opener_kwargs) as ds:
            # Case 1: explicit base_time + lead_time variables
            if 'base_time' in ds.variables and 'lead_time' in ds.variables:
                base_time_val = ds['base_time'].values
                lead_time_vals = ds['lead_time'].values

                if np.issubdtype(base_time_val.dtype, np.datetime64):
                    base_time = pd.to_datetime(base_time_val).to_pydatetime()
                elif isinstance(base_time_val, (str, bytes)):
                    base_time = pd.to_datetime(base_time_val).to_pydatetime()
                else:
                    base_time = None

                if np.issubdtype(lead_time_vals.dtype, np.timedelta64):
                    lead_times = [pd.to_timedelta(lt).to_pytimedelta() for lt in lead_time_vals]
                elif np.issubdtype(lead_time_vals.dtype, np.integer):
                    lead_times = [timedelta(hours=int(lt)) for lt in lead_time_vals]
                else:
                    lead_times = None

                if base_time and lead_times:
                    return file_path, base_time, lead_times, opener_kwargs

            # Case 2: infer from 'time' variable
            if 'time' in ds.variables:
                time_var = ds['time'].values
                if np.issubdtype(time_var.dtype, np.datetime64):
                    base_time = pd.to_datetime(time_var[0]).to_pydatetime()
                    lead_times = [pd.to_datetime(t).to_pydatetime() - base_time for t in time_var]
                    return file_path, base_time, lead_times, opener_kwargs

                elif np.issubdtype(time_var.dtype, (np.timedelta64, np.integer)):
                    candidates = [file_path.stem, file_path.parent.name, file_path.parent.parent.name]
                    best_dt, best_score = None, 0
                    for candidate in candidates:
                        dt, score = try_parse_datetime_from_str(candidate)
                        if dt is not None and score > best_score:
                            best_dt, best_score = dt, score
                    base_time = best_dt
                    if base_time is None:
                        return None
                    if np.issubdtype(time_var.dtype, np.timedelta64):
                        lead_times = [pd.to_timedelta(t).to_pytimedelta() for t in time_var]
                    else:
                        lead_times = [timedelta(hours=int(t)) for t in time_var]
                    return file_path, base_time, lead_times, opener_kwargs

        return None

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def parse_directory(base_dir, start=None, end=None):
    """
    Walks a directory tree and returns a list of all NetCDF files
    successfully parsed. Each entry is (file_path, base_time, lead_times, opener_kwargs).

    Optionally filter files based on start/end strings (YYYYMMDD).
    """
    results = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if not file.endswith(".nc"):
                continue

            path = Path(root) / file

            # Optional quick filtering by start/end date strings
            if start and end:
                try:
                    candidates = [path.stem, path.parent.name, path.parent.parent.name]
                    keep = False
                    for c in candidates:
                        if len(c) >= 8 and c[:8].isdigit():
                            if ((start is None or c[:8] >= start[:8]) and
                                (end is None or c[:8] <= end[:8])):
                                keep = True
                                break
                    if not keep:
                        continue
                except Exception:
                    pass  # ignore filtering errors

            parsed = parse_file(path)
            if parsed is not None:
                results.append(parsed)
    return results
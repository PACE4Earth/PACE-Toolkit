import os
import re
from pathlib import Path
import pandas as pd
import xarray as xr
from datetime import timedelta

# GROUP_NAMES = ["truth", "prediction", "input"]
GROUP_NAMES = ["prediction"]

def parse_file(file_path: Path):
    """
    Parses a single NetCDF file with groups.
    Returns a list of tuples:
        (file_path, base_time, lead_times, opener_kwargs)
    """
    results = []

    # Open root dataset to get the global 'time' variable
    try:
        with xr.open_dataset(file_path, engine="netcdf4") as ds_root:
            if "time" not in ds_root:
                print(f"'time' variable not found in root dataset of {file_path}")
                return results
            
            times = ds_root["time"].values
    except Exception as e:
        print(f"Failed to read root dataset {file_path}: {e}")
        return results

    # For each group, append tuples
    for group in GROUP_NAMES:
        opener_kwargs = {"engine": "netcdf4", "group": group}
        try:
            with xr.open_dataset(file_path, **opener_kwargs) as ds_group:
                # Only include group if it exists
                if ds_group is None or len(ds_group.data_vars) == 0:
                    continue
                
                for t in times:
                    match = re.search(r'(\d+)h', str(file_path))
                    lead_times = [timedelta(hours=int(match.group(1))) if match else timedelta(hours=0)]
                    base_time = pd.to_datetime(t).to_pydatetime()
                    results.append((file_path, base_time, lead_times, opener_kwargs))
        except Exception as e:
            print(e)
            # Skip missing groups
            continue
    
    return results

def parse_directory_groups(base_dir, start=None, end=None):
    """
    Walks a directory tree and returns a list of all NetCDF files
    successfully parsed. Each entry is (file_path, base_time, lead_times, opener_kwargs).
    """
    results = []
    for file in os.listdir(base_dir):
        if not file.endswith(".nc"):
            continue

        path = Path(base_dir) / file
        parsed = parse_file(path)
        if parsed is not None:
            results.extend(parsed)
                
    print(len(results), "files parsed from directory:", base_dir)
    
    # for it in range(len(results)):
    #     print(len(results[it]), "entries in first file")

    return results

if __name__ == "__main__":
    
    path = Path("/p/scratch/hclimrep/pavel1/CorrDiff/CorrDiff_Output_Code4Earth/corrdiff_output_ensemble_18h.nc")
    results = parse_file(path)
    
    for i in range(len(results[:100])):
        print(results[i])
    # print(results[0])
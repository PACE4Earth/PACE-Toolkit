from pathlib import Path
import pandas as pd
import xarray as xr
from datetime import timedelta

GROUP_NAMES = ["truth", "prediction", "input"]

def parse_single_file(file_path: Path):
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
                    base_time = pd.to_datetime(t).to_pydatetime()
                    lead_times = [timedelta(hours=18)]  # single lead time
                    results.append((file_path, base_time, lead_times, opener_kwargs))
        except Exception:
            # Skip missing groups
            continue
    
    return results

# results = parse_single_file("/p/scratch/hclimrep/pavel1/CorrDiff/CorrDiff_Output_Code4Earth/corrdiff_output_ensemble_18h.nc")
# for r in results:
#     print(r)

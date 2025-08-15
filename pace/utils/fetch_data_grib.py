"""
Download GraphCastGFS GRIB2 from AWS Open Data, clip to Europe,
include geopotential on isobaric levels, and save NetCDF.

Bucket layout:
  s3://noaa-nws-graphcastgfs-pds/graphcastgfs.YYYYMMDD/CC/forecasts_13_levels/
    graphcastgfs.tCCz.pgrb2.0p25.fFFF
"""

import argparse
import os
from datetime import datetime
import eccodes
from typing import List, Dict, Any

import numpy as np
import xarray as xr
from botocore import UNSIGNED
from botocore.config import Config
import boto3

BUCKET = "noaa-nws-graphcastgfs-pds"
PRODUCT_DIR = "forecasts_13_levels"
PRODUCT = "pgrb2.0p25"
G0 = 9.80665  # m s^-2

# Surface shortNames + geopotential variants on pressure levels
SURFACE_VARS = {"t2m", "10u", "10v", "msl"}
LEVEL_VARS = {"z", "gh"}  # geopotential (m^2 s^-2) or geopotential height (gpm≈m)


def build_key(date: datetime, cycle: int, fhr: int) -> str:
    ymd = date.strftime("%Y%m%d")
    cc = f"{cycle:02d}"
    fff = f"{fhr:03d}"
    return (
        f"graphcastgfs.{ymd}/{cc}/{PRODUCT_DIR}/"
        f"graphcastgfs.t{cc}z.{PRODUCT}.f{fff}"
    )


def download_files(date: datetime, cycle: int, fhrs: List[int], outdir: str) -> List[str]:
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED, region_name="us-east-1"))
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for fh in fhrs:
        key = build_key(date, cycle, fh)
        dest = os.path.join(outdir, os.path.basename(key))
        print(f"→ s3://{BUCKET}/{key}")
        try:
            s3.download_file(BUCKET, key, dest)
            print(f"  saved {dest} ({os.path.getsize(dest)/1e6:.1f} MB)")
            paths.append(dest)
        except Exception as e:
            print(f"  download failed: {e}")
    return paths


# def open_and_select_vars(grib_path: str) -> xr.Dataset:
#     """
#     Open a GRIB with cfgrib across groups, keep:
#       - surface vars: t2m, 10u, 10v, msl
#       - level vars:   z (geopotential), gh (geopotential height)
#     Merge what we find.
#     """
#     import cfgrib
#     ds_list = cfgrib.open_datasets(grib_path, backend_kwargs={"indexpath": ""})
#     keep = []
#     for ds in ds_list:
#         # pick surface vars present in this group
#         keep_vars = [v for v in ds.data_vars if v in SURFACE_VARS or v in LEVEL_VARS]
#         if keep_vars:
#             keep.append(ds[keep_vars])
#     if not keep:
#         # fallback: at least return something
#         keep = [ds_list[0]]
#     return xr.merge(keep, compat="override")


# Define the variables we want to extract
SURFACE_VARS = ['t2m', '10u', '10v', 'msl']
LEVEL_VARS = ['z', 'gh']
TARGET_VARS = SURFACE_VARS + LEVEL_VARS

def open_and_select_vars(grib_path: str) -> xr.Dataset:
    """
    Opens a GRIB file using the eccodes library, selecting and merging
    specific surface and multi-level variables into a single xarray.Dataset.

    This function manually replicates the variable grouping and building
    process handled by cfgrib.open_datasets.
    
    It keeps:
      - surface vars: t2m, 10u, 10v, msl
      - level vars:   z (geopotential), gh (geopotential height)
    """
    # Intermediate storage for data and metadata
    # For 3D vars, we collect 2D slices and their level coordinates
    # Format: { 'var_name': {'slices': [slice1, slice2], 'levels': [500, 700]} }
    collected_data: Dict[str, Dict[str, Any]] = {}
    
    # Store coordinates once, assuming a consistent grid
    lat_coords, lon_coords = None, None

    with open(grib_path, 'rb') as f:
        while True:
            # Get one GRIB message from the file
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break # End of file

            try:
                short_name = eccodes.codes_get(gid, 'shortName')

                # Skip messages for variables we don't need
                if short_name not in TARGET_VARS:
                    continue

                # Initialize storage for this variable if it's the first time we see it
                if short_name not in collected_data:
                    collected_data[short_name] = {
                        'slices': [], 
                        'levels': [], 
                        'units': eccodes.codes_get(gid, 'units')
                    }

                # --- Extract data and coordinates ---
                nj = eccodes.codes_get(gid, 'Nj')
                ni = eccodes.codes_get(gid, 'Ni')
                values = eccodes.codes_get_values(gid).reshape((nj, ni))
                
                # Store the data slice
                collected_data[short_name]['slices'].append(values)

                # --- Handle dimensions (2D vs 3D) ---
                if short_name in LEVEL_VARS:
                    # For 3D vars, get the vertical level coordinate
                    # Note: You might need to try other keys like 'hybrid'
                    level = eccodes.codes_get(gid, 'isobaricInhPa')
                    collected_data[short_name]['levels'].append(level)
                
                # --- Store grid coordinates (only needed once) ---
                if lat_coords is None:
                    lats = eccodes.codes_get_array(gid, 'latitudes').reshape((nj, ni))
                    lons = eccodes.codes_get_array(gid, 'longitudes').reshape((nj, ni))
                    lat_coords = lats[:, 0]
                    lon_coords = lons[0, :]

            finally:
                # IMPORTANT: Always release the message handle
                if gid:
                    eccodes.codes_release(gid)

    if not collected_data:
        # Return an empty dataset if no target variables were found
        return xr.Dataset()

    # --- Assemble the final xarray.Dataset ---
    data_vars = {}
    ds_coords = {'latitude': lat_coords, 'longitude': lon_coords}
    level_coord_name = 'isobaricInhPa' # Define a standard name for the level coordinate

    for name, data in collected_data.items():
        if name in LEVEL_VARS:
            # --- Process 3D variables ---
            levels = data['levels']
            slices = data['slices']
            
            # CRITICAL: Sort levels and reorder data slices accordingly
            sorted_indices = np.argsort(levels)
            sorted_levels = np.array(levels)[sorted_indices]
            sorted_slices = [slices[i] for i in sorted_indices]
            
            # Stack 2D slices into a single 3D array
            final_3d_data = np.stack(sorted_slices, axis=0)

            # Define the DataArray for this variable
            data_vars[name] = (
                (level_coord_name, "latitude", "longitude"),
                final_3d_data,
                {"units": data['units']}
            )
            # Add the level coordinate to the dataset's coordinates
            if level_coord_name not in ds_coords:
                 ds_coords[level_coord_name] = sorted_levels
        
        elif name in SURFACE_VARS:
            # --- Process 2D variables ---
            # Surface variables have only one slice
            data_vars[name] = (
                ("latitude", "longitude"),
                data['slices'][0],
                {"units": data['units']}
            )

    return xr.Dataset(data_vars, coords=ds_coords)


def clip_region(ds: xr.Dataset, lon_min=-15.0, lon_max=45.0, lat_min=30.0, lat_max=75.0) -> xr.Dataset:
    # Detect coord names
    lon_name = "longitude" if "longitude" in ds.coords else ("lon" if "lon" in ds.coords else None)
    lat_name = "latitude" if "latitude" in ds.coords else ("lat" if "lat" in ds.coords else None)
    if not lon_name or not lat_name:
        raise ValueError("Could not find longitude/latitude coordinates in dataset.")

    # Handle 0..360 vs -180..180
    lon_vals = ds[lon_name].values
    use_0360 = np.nanmax(lon_vals) > 180
    wrap = lambda lam: lam % 360

    if use_0360:
        lmin, lmax = wrap(lon_min), wrap(lon_max)
        if lmax < lmin:
            west = ds.sel({lon_name: slice(lmin, 360)})
            east = ds.sel({lon_name: slice(0, lmax)})
            ds = xr.concat([west, east], dim=lon_name)
        else:
            ds = ds.sel({lon_name: slice(lmin, lmax)})
    else:
        ds = ds.sel({lon_name: slice(lon_min, lon_max)})

    # Latitude may be descending
    lat_vals = ds[lat_name].values
    if lat_vals[0] > lat_vals[-1]:
        ds = ds.sel({lat_name: slice(lat_max, lat_min)})
    else:
        ds = ds.sel({lat_name: slice(lat_min, lat_max)})

    return ds


def select_levels_if_present(ds: xr.Dataset, levels_hpa: List[int]) -> xr.Dataset:
    """
    If an isobaric dimension exists, subset to desired levels (in hPa).
    Accepts both 'isobaricInhPa' and 'isobaricInPa'.
    """
    if "isobaricInhPa" in ds.dims:
        return ds.sel(isobaricInhPa=levels_hpa)
    if "isobaricInPa" in ds.dims:
        # convert hPa→Pa
        return ds.sel(isobaricInPa=[int(l * 100) for l in levels_hpa])
    return ds  # nothing to do for pure-surface datasets


def add_geopotential_height(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure a 'Z' variable (geopotential height, meters) exists on levels.
    If 'gh' is present, use it (assume gpm≈m). If only 'z' present (m^2/s^2),
    convert: Z = z / g0.
    """
    has_level_dim = ("isobaricInhPa" in ds.dims) or ("isobaricInPa" in ds.dims)
    if not has_level_dim:
        return ds

    if "Z" in ds:
        return ds

    if "gh" in ds:
        return ds.assign(Z=ds["gh"].astype("float32"))

    if "z" in ds:
        return ds.assign(Z=(ds["z"] / G0).astype("float32"))

    return ds


def main():
    p = argparse.ArgumentParser(
        description="Download GraphCastGFS GRIB2 from AWS, clip to Europe, include geopotential, save NetCDF."
    )
    p.add_argument("--date", required=True, help="Init date YYYY-MM-DD (UTC)")
    p.add_argument("--cycle", type=int, choices=[0, 6, 12, 18], required=True, help="Cycle hour (UTC)")
    p.add_argument("--fhrs", default="0,6,24", help="Comma-separated forecast hours, e.g. 0,6,24")
    p.add_argument("--levels", default="500", help="Comma-separated pressure levels (hPa) for geopotential, e.g. 500 or 850,500")
    p.add_argument("--outdir", default="graphcastgfs_grib", help="Where to store GRIB2 files")
    p.add_argument("--outfile", default="graphcastgfs_eu.nc", help="Output NetCDF file")
    p.add_argument("--lon-min", type=float, default=-15.0)
    p.add_argument("--lon-max", type=float, default=45.0)
    p.add_argument("--lat-min", type=float, default=30.0)
    p.add_argument("--lat-max", type=float, default=75.0)
    args = p.parse_args()

    date = datetime.strptime(args.date, "%Y-%m-%d")
    fhrs = [int(s) for s in args.fhrs.split(",") if s.strip()]
    levels = [int(s) for s in args.levels.split(",") if s.strip()]

    # 1) Download
    paths = download_files(date, args.cycle, fhrs, args.outdir)
    if not paths:
        print("No files downloaded; exiting.")
        return

    # 2) Open, select vars (surface + geopotential), clip, select levels, add Z
    clipped = []
    for pth in paths:
        ds = open_and_select_vars(pth)
        ds = clip_region(ds, args.lon_min, args.lon_max, args.lat_min, args.lat_max)
        ds = select_levels_if_present(ds, levels)
        ds = add_geopotential_height(ds)

        # unify coord names to lon/lat for convenience
        ren = {}
        if "longitude" in ds.coords: ren["longitude"] = "lon"
        if "latitude" in ds.coords: ren["latitude"] = "lat"
        ds = ds.rename(ren)
        clipped.append(ds)

    # 3) Concatenate along forecast step (cfgrib provides 'step' for fhours)
    if "step" in clipped[0].coords:
        out = xr.concat(clipped, dim="step")
    else:
        out = xr.concat(clipped, dim="time" if "time" in clipped[0].dims else "index")

    # 4) Save compact NetCDF
    enc = {v: dict(zlib=True, complevel=4, shuffle=True, dtype="float32") for v in out.data_vars}
    out.load().to_netcdf(args.outfile, engine="netcdf4", encoding=enc)
    print(f"Saved → {args.outfile}  ({os.path.getsize(args.outfile)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
    
# srun python fetch_data_grib.py --date 2025-08-13 --cycle 12 --fhrs 0,6,24 --levels 500
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
from typing import List

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


def open_and_select_vars(grib_path: str) -> xr.Dataset:
    """
    Open a GRIB with cfgrib across groups, keep:
      - surface vars: t2m, 10u, 10v, msl
      - level vars:   z (geopotential), gh (geopotential height)
    Merge what we find.
    """
    import cfgrib
    ds_list = cfgrib.open_datasets(grib_path, backend_kwargs={"indexpath": ""})
    keep = []
    for ds in ds_list:
        # pick surface vars present in this group
        keep_vars = [v for v in ds.data_vars if v in SURFACE_VARS or v in LEVEL_VARS]
        if keep_vars:
            keep.append(ds[keep_vars])
    if not keep:
        # fallback: at least return something
        keep = [ds_list[0]]
    return xr.merge(keep, compat="override")


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
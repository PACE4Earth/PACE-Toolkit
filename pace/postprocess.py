import os
import json
import numpy as np
from pathlib import Path
import argparse
import time
import xarray as xr

from mpi4py import MPI


def is_viz_enabled(config: dict, key: str) -> bool:
    value = config.get("visualization", {}).get(key, False)
    if isinstance(value, dict):
        return value.get("enabled", False)
    return bool(value)


def open_xarray_zarr(zarr_path):
    """
    Open an xarray-compatible Zarr dataset lazily.

    Returns:
        var_store: dict { var_name: DataArray (lazy) }
        coords: dict of coordinates (numpy arrays)
        ds: full xarray.Dataset (lazy)
    """
    ds = xr.open_zarr(zarr_path)  # lazy loading
    var_store = {var: ds[var] for var in ds.data_vars}

    coords = {name: ds.coords[name].values for name in ['lat', 'lon', 'level', 'base_time', 'lead_time'] if name in ds.coords}

    return var_store, coords, ds


def select_sample_leadtimes(ds, max_leadtimes: int = 5):
    """
    Select up to max_leadtimes evenly spaced lead times from the dataset.
    """
    all_leadtimes = np.array(ds['lead_time'].values, dtype='timedelta64[h]').astype(int)
    all_leadtimes = np.unique(all_leadtimes)
    if len(all_leadtimes) <= max_leadtimes:
        return all_leadtimes.tolist()
    idxs = np.linspace(0, len(all_leadtimes) - 1, max_leadtimes, dtype=int)
    return [all_leadtimes[i] for i in idxs]


def main():
    
    comm = MPI.COMM_WORLD
    
    if comm.Get_rank() != 0:
        return
    
    time_start = time.perf_counter()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_CONFIG_PATH = Path(os.path.join(BASE_DIR, 'configs', 'config_template.json'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config file (overrides env var and default)")
    args = parser.parse_args()

    if args.config is not None:
        config_path = Path(args.config)
    elif "CONFIG_PATH" in os.environ:
        config_path = Path(os.environ["CONFIG_PATH"])
    else:
        config_path = DEFAULT_CONFIG_PATH

    config_path = config_path.expanduser().resolve()
    with open(config_path, "r") as f:
        config = json.load(f)

    try:
        outputs_dir = Path(os.environ['OUTPUTS_DIR_PATH'])
    except:
        outputs_dir = Path(os.path.expandvars(config.get("outputs_dir", "")))
    if not outputs_dir.exists():
        outputs_dir = Path(__file__).resolve().parent / "outputs"
    
    try:
        plots_dir = Path(os.environ['PLOTS_DIR_PATH'])
    except:
        plots_dir = Path(os.path.expandvars(str(config["visualization"].get("plots_dir", ""))))
    if not plots_dir.exists():
        plots_dir = Path(__file__).resolve().parent / "plots"

    # --- MODEL dataset ---
    model_name = config['datasets']['model']['name']
    model_path = outputs_dir / f"{model_name}.zarr"
    model_store, coords, model_ds = open_xarray_zarr(model_path)
    model_leadtimes = select_sample_leadtimes(model_ds, max_leadtimes=5)

    # --- REFERENCE dataset ---
    ref_name = None
    ref_store = {}
    ref_ds = None
    if "reference" in config["datasets"] and config["datasets"]["reference"].get("name"):
        ref_name = config['datasets']['reference']['name']
        ref_path = outputs_dir / f"{ref_name}.zarr"
        ref_store, _, ref_ds = open_xarray_zarr(ref_path)

    # --- HISTOGRAM visualization ---
    try:
        if is_viz_enabled(config, "histogram"):
            print("Running histogram visualization...")
            from utils.plot_utils import histogram  
            bin_config = histogram.BIN_CONFIG

            _, model_hist = histogram.compute_histograms(
                model_store, selected_leadtimes=model_leadtimes, bin_config=bin_config
            )

            _, ref_hist = histogram.compute_histograms(
                ref_store, selected_leadtimes=None, bin_config=bin_config
            ) if ref_store else ({}, {})

            histogram.plot_hist(model_hist, ref_hist, plots_dir, bin_config, model_name, ref_name)
        else:
            print("Histogram visualization disabled in config.\n")
    except Exception as e:
        print(e)

    # --- VERTICAL PROFILE visualization ---
    try:
        if is_viz_enabled(config, "vertical_profile"):
            print("\nRunning vertical profile visualization...")
            from utils.plot_utils import vertical_profile

            summary_stats = config.get("visualization", {}).get("summary_stats", ["mean"])

            profiles = vertical_profile.compute_summary_stats(
                model_store,
                coords['lat'],
                selected_leadtimes=model_leadtimes,
                summary_stats=summary_stats,
            )
            ref_profiles = vertical_profile.compute_summary_stats(
                ref_store,
                coords['lat'] if coords.get('lat') is not None else None,
                selected_leadtimes=None,
                summary_stats=summary_stats,
            ) if ref_store else {}

            vertical_profile.plot_profiles(
                profiles,
                ref_profiles,
                coords.get('level'),
                plots_dir,
                summary_stats,
                model_name,
                ref_name,
                model_leadtimes
            )
        else:
            print("Vertical profile visualization disabled in config.\n")
    except Exception as e:
        print(e)
      
    # --- CORRELATION & BIVAR. HIST. visualization ---  
    try:
        if is_viz_enabled(config, "correlation"):
            print("\nRunning correlation visualization...")
            
            from utils.plot_utils import correlation
            
            correlation.plot_corr_lt(
                model_ds=model_ds,
                ref_ds=ref_ds,
                model_name=model_name,
                ref_name=ref_name,
                plots_dir=plots_dir,
            )
            
            correlation.plot_corr_time_series(
                model_ds=model_ds,
                ref_ds=ref_ds,
                model_name=model_name,
                ref_name=ref_name,
                plots_dir=plots_dir,
            )
            
            valid_times = np.unique(ref_ds['base_time'].values + ref_ds['lead_time'].values)
            
            correlation.plot_bivar_hist(
                model_name=model_name,
                ref_name=ref_name,
                outputs_dir=outputs_dir,
                plots_dir=plots_dir,
                valid_times=valid_times,
            )
            
        else:
            print("Correlation visualization disabled in config.\n")
    except Exception as e:
        print(e)
        
    # --- CORRELATION MAP visualization ---    
    try:
        if is_viz_enabled(config, "correlation_map"):
            print("\nRunning correlation map visualization...")
            
            from utils.plot_utils import correlation_map
            
            correlation_map.plot_map(
                model_ds=model_ds,
                ref_ds=ref_ds,
                model_name=model_name,
                ref_name=ref_name,
                plots_dir=plots_dir,
                coords=coords,
            )
            
        else:
            print("Correlation map visualization disabled in config.\n")
    except Exception as e:
        print(e)
        
    # --- SPATIAL SLICE visualization ---
    try:
        if is_viz_enabled(config, "spatial_slice"):
            from utils.plot_utils import spatial_slice
            print("\nRunning spatial slice visualization...")

            spatial_cfg = config["visualization"]["spatial_slice"]
            variable = spatial_cfg.get("variable", "")
            level = spatial_cfg.get("level", 850)  # hPa
            samples = spatial_cfg.get("samples", 1)

            spatial_slice.plot_spatial_slice(
                model_store,
                coords,
                variable=variable,
                level=level,
                samples=samples,
                save_dir=str(plots_dir),
                dataset_name=model_name,
            )

            if ref_store:
                spatial_slice.plot_spatial_slice(
                    ref_store,
                    coords,
                    variable=variable,
                    level=level,
                    samples=samples,
                    save_dir=str(plots_dir),
                    dataset_name=ref_name,
                )
        else:
            print("Spatial slice visualization disabled in config.\n")

    except Exception as e:
        print(e)


    # --- SPATIAL AVERAGE visualization ---
    try:
        if is_viz_enabled(config, "spatial_average"):
            from utils.plot_utils import spatial_average
            print("\nRunning spatial average visualization...")

            spatial_average.plot_spatial_averages(
                model_store,
                coords,
                save_dir=str(plots_dir),
                dataset_name=model_name,
            )

            if ref_store:
                spatial_average.plot_spatial_averages(
                    ref_store,
                    coords,
                    save_dir=str(plots_dir),
                    dataset_name=ref_name,
                )
        else:
            print("Spatial average visualization disabled in config.\n")

    except Exception as e:
        print(e)

    # --- TIME SERIES visualization ---
    try:
        if is_viz_enabled(config, "time_series"):
            print("Running time series visualization...")
            from utils.plot_utils import time_series

            # Prepare model
            model_ts = time_series.prepare_time_series(
                model_store, selected_leadtimes=model_leadtimes
            )

            # Prepare reference (all lead times)
            ref_ts = time_series.prepare_time_series(
                ref_store, selected_leadtimes=None
            ) if ref_store else None

            time_series.plot_time_series(
                model_ts, ref_ts, plots_dir, model_name=model_name, ref_name=ref_name
            )
        else:
            print("Time series visualization disabled in config.\n")
    except Exception as e:
        print(e)


    time_end = time.perf_counter()
    print(f"\nElapsed time: {time_end - time_start:.2f} s")


if __name__ == "__main__":
    main()

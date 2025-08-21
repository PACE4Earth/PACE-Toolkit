import os
import json
import numpy as np
from pathlib import Path
import time
import zarr  

def is_viz_enabled(config: dict, key: str) -> bool:
    """
    Check if a visualization type is enabled in config.
    Example: is_viz_enabled(config, "histogram")
    """
    return bool(config.get("visualization", {}).get(key, False))


def unpack_custom_zarr_vars(zarr_root):
    """
    Return a dictionary:
    { (var_name, base_time, lead_time): zarr_array }
    """
    zarr_root = Path(zarr_root)
    store = {}

    for base_dir in zarr_root.iterdir():
        if not base_dir.is_dir():
            continue
        try:
            base_time = np.datetime64(base_dir.name)
        except Exception:
            continue

        for lead_dir in base_dir.iterdir():
            if not lead_dir.is_dir():
                continue
            try:
                lead_time = int(lead_dir.name.replace("h", ""))
            except ValueError:
                continue

            for var_dir in lead_dir.iterdir():
                if not var_dir.is_dir() or var_dir.name.startswith('.'):
                    continue

                arr = zarr.open_array(str(var_dir), mode='r')
                store[(var_dir.name, base_time, lead_time)] = arr

    return store

def load_coords_from_zarr(zarr_root_path):
    zarr_root_path = Path(zarr_root_path)
    coords = {}
    for coord_name in ['lat', 'lon', 'pressure_levels']:
        coord_path = zarr_root_path / coord_name
        if coord_path.exists():
            coords[coord_name] = zarr.open_array(str(coord_path), mode='r')[:]
        else:
            print(f"Coordinate {coord_name} not found at {coord_path}")

    return coords

def select_sample_leadtimes(zarr_path: Path, total_expected: int, max_leadtimes: int = 5) -> list[int]:
    """
    Select up to max_leadtimes evenly spaced lead times for the model dataset.
    Reads across base_time directories but stops once total_expected unique lead times are found.
    """
    zarr_path = Path(zarr_path)
    found_leadtimes = set()

    for base_dir in zarr_path.iterdir():
        if not base_dir.is_dir():
            continue
        for lead_dir in base_dir.iterdir():
            if not lead_dir.is_dir():
                continue
            try:
                lead_time = int(lead_dir.name.replace("h", ""))
                found_leadtimes.add(lead_time)
            except ValueError:
                continue

            if len(found_leadtimes) >= total_expected:
                break

        if len(found_leadtimes) >= total_expected:
            break

    found_leadtimes = sorted(found_leadtimes)
    if len(found_leadtimes) <= max_leadtimes:
        return found_leadtimes

    idxs = np.linspace(0, len(found_leadtimes) - 1, max_leadtimes, dtype=int)
    return [found_leadtimes[i] for i in idxs]


def main():
    time_start = time.perf_counter()
    config_path = Path(__file__).resolve().parent / "configs" / "config_graphcast.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    outputs_dir = Path(os.path.expandvars(config.get("outputs_dir", "")))
    if not outputs_dir.exists():
        outputs_dir = Path(__file__).resolve().parent / "outputs"
    
    plots_dir = Path(os.path.expandvars(str(config["visualization"].get("plots_dir"))))
    if not outputs_dir.exists():
        plots_dir = Path(__file__).resolve().parent / "plots"
    # plots_dir = Path(os.path.expandvars(str(config["visualization"].get("plots_dir") or Path(os.path.abspath(os.path.dirname(__file__))) / "plots")))

    model_name = config['datasets']['model']['name']
    total_leadtimes = config["time"]["num_lead_times"]
    model_path = outputs_dir / f"{model_name}.zarr"
    model_leadtimes = select_sample_leadtimes(model_path, total_expected=total_leadtimes)
    model_store = unpack_custom_zarr_vars(model_path)

    coords = load_coords_from_zarr(model_path)

    lats = coords.get("lat")
    lons = coords.get("lon")
    pressure_levels = coords.get("pressure_levels")

    ref_name = None
    ref_store = {}
    if "reference" in config["datasets"] and config["datasets"]["reference"].get("name"):
        ref_name = config['datasets']['reference']['name']
        ref_path = outputs_dir / f"{ref_name}.zarr"
        ref_store = unpack_custom_zarr_vars(ref_path)

    # --- HISTOGRAM visualization ---
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

    # --- VERTICAL PROFILE visualization ---
    if is_viz_enabled(config, "vertical_profile"):
        print("\nRunning vertical profile visualization...")
        from utils.plot_utils import vertical_profile

        summary_stats = config.get("visualization", {}).get("summary_stats", ["mean"])

        profiles = vertical_profile.compute_summary_stats(
            model_store,
            lats,
            selected_leadtimes=model_leadtimes,
            summary_stats=summary_stats,
        )
        ref_profiles = vertical_profile.compute_summary_stats(
            ref_store,
            lats,
            selected_leadtimes=None,
            summary_stats=summary_stats,
        ) if ref_store else {}

        vertical_profile.plot_profiles(
            profiles,
            ref_profiles,
            pressure_levels,
            plots_dir,
            summary_stats,
            model_name,
            ref_name,
            model_leadtimes
        )
    else:
        print("Vertical profile visualization disabled in config.\n")

    if is_viz_enabled(config, "spatial_slice"):
        from utils.plot_utils import spatial_slice
        print("\nRunning spatial slice visualization...")

        spatial_cfg = config["visualization"]["spatial_slice"]
        variable = spatial_cfg.get("variable", "temperature")
        level = spatial_cfg.get("level", 850)  # hPa
        samples = spatial_cfg.get("samples", 1)
        geopotential_level = spatial_cfg.get("geopotential_level", None)

        # Plot model slices
        spatial_slice.plot_spatial_slice(
            model_store,
            coords,
            variable=variable,
            level=level,
            samples=samples,
            geopotential=model_store if "geopotential" in [k[0] for k in model_store.keys()] else None,
            geopotential_level=geopotential_level,
            save_dir=str(plots_dir),
            dataset_name=model_name,
        )

        # Plot reference slices if available
        if ref_store:
            spatial_slice.plot_spatial_slice(
                ref_store,
                coords,
                variable=variable,
                level=level,
                samples=samples,
                geopotential=ref_store if "geopotential" in [k[0] for k in ref_store.keys()] else None,
                geopotential_level=geopotential_level,
                save_dir=str(plots_dir),
                dataset_name=ref_name,
            )
    else:
        print("Spatial slice visualization disabled in config.\n")


    time_end = time.perf_counter()
    print(f"\nElapsed time: {time_end - time_start:.2f} s")

if __name__ == "__main__":
    main()

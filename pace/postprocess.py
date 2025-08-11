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
    # Try opening latitude, longitude, and level arrays from the root
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
    config_path = Path(__file__).resolve().parent / "configs" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    outputs_dir = Path(os.path.expandvars(config["outputs_dir"]))
    plots_dir = Path(os.path.expandvars(str(config["visualization"].get("plots_dir") or Path(os.path.abspath(os.path.dirname(__file__))) / "plots")))

    model_name = config['datasets']['model']['name']
    total_leadtimes = config["time"]["lead_times"]
    model_path = outputs_dir / f"{model_name}.zarr"
    model_leadtimes = select_sample_leadtimes(model_path, total_expected=total_leadtimes)
    model_store = unpack_custom_zarr_vars(model_path)

    coords = load_coords_from_zarr(model_path)

    print("Latitude values:", coords.get("lat"))
    print("Longitude values:", coords.get("lon"))
    print("Level values:", coords.get("pressure_levels"))


    ref_name = None
    ref_store = {}
    if "reference" in config["datasets"] and config["datasets"]["reference"].get("name"):
        ref_name = config['datasets']['reference']['name']
        ref_path = outputs_dir / f"{ref_name}.zarr"
        ref_store = unpack_custom_zarr_vars(ref_path)

    # --- HISTOGRAM visualization ---
    if is_viz_enabled(config, "histogram"):
        print("Running histogram visualization...")
        from utils.plot_utils import histogram  # only import if needed

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
            selected_leadtimes=model_leadtimes,
            summary_stats=summary_stats,
        )
        ref_profiles = vertical_profile.compute_summary_stats(
            ref_store,
            selected_leadtimes=None,
            summary_stats=summary_stats,
        ) if ref_store else {}

        vertical_profile.plot_profiles(
            profiles,
            ref_profiles,
            plots_dir,
            summary_stats,
            model_name,
            ref_name,
        )
    else:
        print("Vertical profile visualization disabled in config.\n")

    time_end = time.perf_counter()
    print(f"\nElapsed time: {time_end - time_start:.2f} s")

if __name__ == "__main__":
    main()

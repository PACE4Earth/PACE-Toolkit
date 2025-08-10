import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import time
import zarr
from typing import List, Dict, Tuple


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


def select_sample_leadtimes(zarr_path: Path, total_expected: int, max_leadtimes: int = 5) -> List[int]:
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


def get_bins_for_variable(var_name: str, vmin: float, vmax: float, bins: int, scale: str):
    """Return bin edges array based on scale type."""
    if scale == "log":
        epsilon = 1e-10  # small offset to avoid log(0)
        if vmin <= 0:
            vmin = epsilon  # force positive min for log scale
        bins_edges = np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)
    else:
        bins_edges = np.linspace(vmin, vmax, bins + 1)
    return bins_edges


def compute_histograms(
    zarr_path: Path,
    selected_leadtimes: List[int] = None,
    bins: int = 100,
    bin_config: Dict[str, Dict] = None,
):
    """
    Compute histograms only for variables defined in bin_config,
    using their predefined vmin, vmax, and scale.
    Variables not in bin_config are ignored.
    """
    if bin_config is None:
        raise ValueError("bin_config must be provided with variable bin specifications")

    store = unpack_custom_zarr_vars(zarr_path)

    # Prepare histogram storage
    hist_data: Dict[str, Dict[int, np.ndarray]] = {
        var: {} for var in bin_config.keys()
    }

    # Precompute bin edges for each variable
    bin_edges_map = {}
    for var_name, cfg in bin_config.items():
        vmin = cfg.get("vmin")
        vmax = cfg.get("vmax")
        scale = cfg.get("scale", "linear")
        bin_edges_map[var_name] = get_bins_for_variable(var_name, vmin, vmax, bins, scale)

    # Accumulate counts
    for (var_name, _, lead_time), arr in store.items():
        if selected_leadtimes is not None and lead_time not in selected_leadtimes:
            continue

        if var_name not in bin_config:
            # Skip vars not in config
            continue

        bin_edges = bin_edges_map[var_name]

        data = arr[:]

        mask = ~np.isnan(data)
        if not np.any(mask):
            continue
        values = data[mask]

        counts, _ = np.histogram(values, bins=bin_edges)
        if lead_time in hist_data[var_name]:
            hist_data[var_name][lead_time] += counts
        else:
            hist_data[var_name][lead_time] = counts

    # Normalize & store bin centers
    hist_result: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for var_name in hist_data:
        bin_edges = bin_edges_map[var_name]
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        hist_result[var_name] = {}
        for lt, counts in hist_data[var_name].items():
            total = counts.sum()
            hist_result[var_name][lt] = (
                centers,
                counts / total if total > 0 else counts,
            )

    return bin_config, hist_result


def plot_hist(model_hist, ref_hist, output_dir: Path, bin_config: Dict[str, Dict] = None):
    out_dir = Path(output_dir) / "histograms"
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in model_hist.keys():
        plt.figure()
        for lt, (centers, counts) in sorted(model_hist[metric].items(), key=lambda x: x[0]):
            plt.plot(centers, counts, label=f"Model {lt}h")

        if metric in ref_hist:
            total_counts = None
            centers = None
            for _, (c, counts) in ref_hist[metric].items():
                centers = c
                total_counts = counts if total_counts is None else total_counts + counts
            if total_counts is not None and total_counts.sum() > 0:
                total_counts = total_counts / total_counts.sum()
            plt.plot(centers, total_counts, label="Reference", linewidth=2, linestyle="--")

        plt.title(f"Histogram - {metric}")
        plt.xlabel(metric)
        plt.ylabel("Probability")
        plt.legend()
        plt.grid(True)

        # Use log scale x axis if requested
        if bin_config and metric in bin_config:
            if bin_config[metric].get("scale", "linear") == "log":
                plt.xscale("log")

        plt.savefig(out_dir / f"{metric}.png")
        print(f"Saved: {out_dir}/{metric}.png")
        plt.close()


def main():
    time_start = time.perf_counter()
    config_path = Path(__file__).resolve().parent / "configs" / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    bin_config = {
        "geostrophic_wind_ratio": {"vmin": 0, "vmax": 2, "scale": "linear"},
        "hydrostatic_abs_error": {"vmin": 1e-3, "vmax": 1e4, "scale": "log"},
        "hydrostatic_rel_error": {"vmin": 1e-3, "vmax": 1e0, "scale": "log"},
        "hydrostatic_rmse": {"vmin": 1e-3, "vmax": 1e4, "scale": "log"},
        "relative_humidity": {"vmin": 2, "vmax": 100, "scale": "linear"},
        "potential_vorticity": {"vmin": 1e-1, "vmax": 1e1, "scale": "log"},
        # Add more variables here as needed
    }

    outputs_dir = Path(os.path.expandvars(config["outputs_dir"]))
    plots_dir = Path(os.path.expandvars(config["visualization"]["plots_dir"]))
    total_leadtimes = config["time"]["lead_times"]

    model_path = outputs_dir / f"{config['datasets']['model']['name']}.zarr"
    model_leadtimes = select_sample_leadtimes(model_path, total_expected=total_leadtimes)

    _, model_hist = compute_histograms(
        model_path, selected_leadtimes=model_leadtimes, bin_config=bin_config
    )

    ref_hist = {}
    if "reference" in config["datasets"] and config["datasets"]["reference"].get("name"):
        ref_path = outputs_dir / f"{config['datasets']['reference']['name']}.zarr"
        _, ref_hist = compute_histograms(ref_path, bin_config=bin_config)

    plot_hist(model_hist, ref_hist, plots_dir, bin_config=bin_config)

    time_end = time.perf_counter()
    print(f"Elapsed time: {time_end - time_start:.2f} s")


if __name__ == "__main__":
    main()

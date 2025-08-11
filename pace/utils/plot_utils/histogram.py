import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns

BIN_CONFIG = {
    "geostrophic_wind_ratio": {"vmin": 0, "vmax": 2, "scale": "linear"},
    "hydrostatic_abs_error": {"vmin": 1e-3, "vmax": 1e4, "scale": "log"},
    "hydrostatic_rel_error": {"vmin": 1e-3, "vmax": 1e0, "scale": "log"},
    "hydrostatic_rmse": {"vmin": 1e1, "vmax": 1e4, "scale": "log"},
    "relative_humidity": {"vmin": 5, "vmax": 105, "scale": "linear"},
    "potential_vorticity": {"vmin": 1e-1, "vmax": 1e1, "scale": "log"},
    # Add more variables here as needed
}


def get_bins_for_variable(
    var_name: str, vmin: float, vmax: float, bins: int, scale: str
) -> np.ndarray:
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
    store: Dict[Tuple[str, np.datetime64, int], np.ndarray],
    selected_leadtimes: Optional[List[int]] = None,
    bins: int = 100,
    bin_config: Dict[str, Dict] = None,
) -> Tuple[Dict[str, Dict], Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Compute histograms only for variables defined in bin_config,
    using their predefined vmin, vmax, and scale.
    Variables not in bin_config are ignored.

    Returns: bin_config, hist_result
    """
    if bin_config is None:
        raise ValueError("bin_config must be provided with variable bin specifications")

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


def plot_hist(
    model_hist: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    ref_hist: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
    bin_config: Dict[str, Dict] = None,
    model_name: str = "Model",
    ref_name: str = "Reference"
):
    sns.set_style("whitegrid")
    out_dir = output_dir / "histograms"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Define a nice color palette for model lines
    palette = sns.color_palette("tab10", n_colors=len(next(iter(model_hist.values()))))

    for metric in model_hist.keys():
        plt.figure(figsize=(8, 6))

        # Plot model histograms with colors
        for i, (lt, (centers, counts)) in enumerate(sorted(model_hist[metric].items(), key=lambda x: x[0])):
            plt.plot(
                centers, counts,
                label=f"{model_name}: Lt {lt}h",
                color=palette[i % len(palette)],
                linewidth=1.8,
            )

        # Plot reference histogram aggregated over lead times
        if metric in ref_hist:
            total_counts = None
            centers = None
            for _, (c, counts) in ref_hist[metric].items():
                centers = c
                total_counts = counts if total_counts is None else total_counts + counts
            if total_counts is not None and total_counts.sum() > 0:
                total_counts = total_counts / total_counts.sum()
            plt.plot(
                centers, total_counts,
                label=f"{ref_name}",
                linewidth=2.5,
                linestyle="--",
                color='black'
            )

        plt.title(f"Histogram - {metric.replace('_', ' ').capitalize()}", fontsize=16, weight='bold')
        plt.xlabel(f"{metric.replace('_', ' ').capitalize()}", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)

        # Improve axes aesthetics
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6, width=1.2)
        ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')

        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1)

        # Set x scale if specified
        if bin_config and metric in bin_config:
            if bin_config[metric].get("scale", "linear") == "log":
                plt.xscale("log")

        # Legend styling
        leg = plt.legend(frameon=True, fontsize=12, loc='best', edgecolor='black', fancybox=True)
        leg.get_frame().set_alpha(0.9)

        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=300)
        print(f"Saved: {out_dir}/{metric}.png")
        plt.close()

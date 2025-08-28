import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns
import xarray as xr

# Configuration for histogram binning of each variable.
# Each entry specifies the min/max values and whether to use linear or log scaling.
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
    """
    Generate bin edges for histogram computation.

    Parameters
    ----------
    var_name : str
        Name of the variable.
    vmin : float
        Minimum value of the range.
    vmax : float
        Maximum value of the range.
    bins : int
        Number of bins to create.
    scale : str
        Scale type: "linear" or "log".

    Returns
    -------
    np.ndarray
        Array of bin edges.
    """
    if scale == "log":
        epsilon = 1e-10  # small offset to avoid log(0)
        if vmin <= 0:
            vmin = epsilon  # force positive min for log scale
        bins_edges = np.logspace(np.log10(vmin), np.log10(vmax), bins + 1)
    else:
        bins_edges = np.linspace(vmin, vmax, bins + 1)
    return bins_edges


def compute_histograms(
    store: Dict[str, xr.DataArray],
    selected_leadtimes: Optional[List[int]] = None,
    bins: int = 100,
    bin_config: Dict[str, Dict] = None,
) -> Tuple[Dict[str, Dict], Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Compute normalized histograms for variables defined in bin_config and present in store.

    Parameters
    ----------
    store : dict
        Dictionary mapping variable names to xarray.DataArray objects.
        Each DataArray may include coordinate `lead_time` (aligned along dimension `idx`).
    selected_leadtimes : list of int, optional
        Specific lead times (in hours) to compute histograms for.
        If None, all available lead times are used.
    bins : int
        Number of histogram bins.
    bin_config : dict
        Configuration mapping variable names to {"vmin", "vmax", "scale"}.

    Returns
    -------
    filtered_bin_config : dict
        Subset of bin_config containing only variables present in store.
    hist_result : dict
        {var_name: {lead_time: (bin_centers, normalized_counts)}}
    """
    if bin_config is None:
        raise ValueError("bin_config must be provided with variable bin specifications")

    # Precompute bin edges for all variables
    bin_edges_map = {
        var_name: get_bins_for_variable(
            var_name,
            cfg.get("vmin"),
            cfg.get("vmax"),
            bins,
            cfg.get("scale", "linear"),
        )
        for var_name, cfg in bin_config.items()
    }

    hist_data: Dict[str, Dict[int, np.ndarray]] = {}

    for var_name, arr in store.items():
        if var_name not in bin_config:
            continue

        if ("idx" in arr.dims) and ("lead_time" in arr.coords):
            # Case: data with lead_time coordinate
            lt_hours_all = np.array(arr.coords["lead_time"].values, dtype="timedelta64[h]").astype(int)

            # Select lead times
            if selected_leadtimes is None:
                leadtimes_to_use = np.unique(lt_hours_all)
            else:
                leadtimes_to_use = [lt for lt in selected_leadtimes if np.any(lt_hours_all == lt)]

            for lt_hours in leadtimes_to_use:
                idx_mask_np = (lt_hours_all == lt_hours)
                if not np.any(idx_mask_np):
                    continue

                # Subset values for this lead time
                idx_mask = xr.DataArray(idx_mask_np, dims=("idx",), coords={"idx": arr["idx"]})
                subset = arr.where(idx_mask, drop=True).values

                mask = ~np.isnan(subset)
                if not np.any(mask):
                    continue

                values = subset[mask].ravel()
                counts, _ = np.histogram(values, bins=bin_edges_map[var_name])

                # Accumulate counts
                if var_name not in hist_data:
                    hist_data[var_name] = {}
                if lt_hours in hist_data[var_name]:
                    hist_data[var_name][lt_hours] += counts
                else:
                    hist_data[var_name][lt_hours] = counts
        else:
            # Case: no lead_time coordinate → aggregate all values
            data = arr.values
            mask = ~np.isnan(data)
            if not np.any(mask):
                continue

            values = data[mask].ravel()
            counts, _ = np.histogram(values, bins=bin_edges_map[var_name])

            if var_name not in hist_data:
                hist_data[var_name] = {}
            hist_data[var_name][0] = counts  # dummy lead time (0)

    # Normalize histograms and compute bin centers
    hist_result: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    filtered_bin_config: Dict[str, Dict] = {}

    for var_name, lt_dict in hist_data.items():
        bin_edges = bin_edges_map[var_name]
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # midpoints
        hist_result[var_name] = {}
        for lt, counts in lt_dict.items():
            total = counts.sum()
            hist_result[var_name][lt] = (
                centers,
                counts / total if total > 0 else counts,
            )
        filtered_bin_config[var_name] = bin_config[var_name]

    return filtered_bin_config, hist_result


def plot_hist(
    model_hist: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    ref_hist: Dict[str, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    output_dir: Path,
    bin_config: Dict[str, Dict] = None,
    model_name: str = "Model",
    ref_name: str = "Reference"
):
    """
    Plot and save histograms for model and reference data.

    Parameters
    ----------
    model_hist : dict
        Histogram results for the model
        {var_name: {lead_time: (centers, normalized_counts)}}.
    ref_hist : dict
        Histogram results for the reference data
        {var_name: {lead_time: (centers, normalized_counts)}}.
    output_dir : Path
        Directory where plots will be saved (inside a "histograms" subfolder).
    bin_config : dict, optional
        Configuration of variables (used to determine log/linear axis scaling).
    model_name : str
        Label for the model in the legend.
    ref_name : str
        Label for the reference in the legend.
    """
    sns.set_style("whitegrid")
    out_dir = output_dir / "histograms"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Choose color palette: one color per lead time
    palette = sns.color_palette("tab10", n_colors=len(next(iter(model_hist.values()))))

    for metric in model_hist.keys():
        plt.figure(figsize=(8, 6))

        # Plot model histograms, separated by lead time
        for i, (lt, (centers, counts)) in enumerate(sorted(model_hist[metric].items(), key=lambda x: x[0])):
            plt.plot(
                centers, counts,
                label=f"{model_name}: Lt {lt}h",
                linewidth=1.8,
                color=palette[i % len(palette)],
            )

        # Plot reference histogram as an aggregate (summed over lead times)
        if metric in ref_hist and ref_hist[metric]:
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
                    color="black",
                )

        # Axis labels and formatting
        plt.title(f"Histogram - {metric.replace('_', ' ').capitalize()}", fontsize=16, weight="bold")
        plt.xlabel(f"{metric.replace('_', ' ').capitalize()}", fontsize=14)
        plt.ylabel("Probability Density", fontsize=14)

        ax = plt.gca()
        ax.tick_params(axis="both", which="major", labelsize=12, direction="in", length=6, width=1.2)
        ax.tick_params(axis="both", which="minor", direction="in", length=3, width=1)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("black")

        plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=1)

        # Apply log scale if requested
        if bin_config and metric in bin_config:
            if bin_config[metric].get("scale", "linear") == "log":
                plt.xscale("log")

        # Legend styling
        leg = plt.legend(frameon=True, fontsize=12, loc="best", edgecolor="black", fancybox=True)
        leg.get_frame().set_alpha(0.9)

        # Save figure
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}.png", dpi=300)
        print(f"Saved: {out_dir}/{metric}.png")
        plt.close()

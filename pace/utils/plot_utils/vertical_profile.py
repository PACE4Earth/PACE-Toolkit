import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns
import xarray as xr

PROFILE_CONFIG = {
    "geostrophic_wind_ratio": {
        "mean": {"xmin": 0, "xmax": 1, "scale": "linear"},
        "stdev": {"xmin": 0, "xmax": 1, "scale": "linear"},
        "min": {"xmin": 0, "xmax": 0.3, "scale": "linear"},
        "max": {"xmin": 0, "xmax": 50, "scale": "linear"},
    },
    "hydrostatic_abs_error": {
        "mean": {"xmin": 1e-1, "xmax": 1e3, "scale": "log"},
        "stdev": {"xmin": 1e-1, "xmax": 1e3, "scale": "log"},
        "max": {"xmin": 1e0, "xmax": 1e4, "scale": "log"},
        "min": {"xmin": 1e-6, "xmax": 1e-2, "scale": "log"},
    },
    "hydrostatic_rel_error": {
        "mean": {"xmin": 1e-4, "xmax": 1e-1, "scale": "log"},
        "stdev": {"xmin": 1e-4, "xmax": 1e-1, "scale": "log"},
        "max": {"xmin": 1e-3, "xmax": 0.5, "scale": "log"},
        "min": {"xmin": 1e-10, "xmax": 1e-6, "scale": "log"},
    },
    "relative_humidity": {
        "mean": {"xmin": 0, "xmax": 80, "scale": "linear"},
        "stdev": {"xmin": 0, "xmax": 105, "scale": "linear"},
        "min": {"xmin": -10, "xmax": 110, "scale": "linear"},
        "max": {"xmin": 0, "xmax": 120, "scale": "linear"},
    },
    "potential_vorticity": {
        "mean": {"xmin": -5, "xmax": 20, "scale": "linear"},
        "stdev": {"xmin": -5, "xmax": 20, "scale": "linear"},
        "max": {"xmin": -5, "xmax": 20, "scale": "linear"},
        "min": {"xmin": -5, "xmax": 20, "scale": "linear"},
    },
    "total_energy": {
        "mean": {"xmin": 300000, "xmax": 400000, "scale": "linear"},
        "stdev": {"xmin": 0, "xmax": 15000, "scale": "linear"},
        "max": {"xmin": 300000, "xmax": 400000, "scale": "linear"},
        "min": {"xmin": 280000, "xmax": 380000, "scale": "linear"},
    },
    # Add more variables here with per-stat configs as needed
}

use_weights = True  # Set to False to disable latitude weighting

def compute_summary_stats(
    store: Dict[str, xr.DataArray],
    latitudes: np.ndarray,
    selected_leadtimes: Optional[List[int]] = None,
    summary_stats: List[str] = ["mean", "stdev", "min", "max"],
    use_weights: bool = True,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute weighted/unweighted summary statistics over lat/lon per vertical level.

    Parameters:
    - store: dict[var_name: DataArray], shape [idx, level, lat, lon], with coords 'lead_time' (timedelta64[h])
    - latitudes: 1D array of latitudes corresponding to lat dimension
    - selected_leadtimes: list of lead times for model; None means "reference" → collapse all leadtimes
    - summary_stats: list of stats to compute: mean, stdev, min, max
    - use_weights: whether to apply latitude weighting

    Returns:
    - results: dict[var_name][stat] =
        - np.ndarray shape [num_leadtimes, num_levels] for model
        - np.ndarray shape [num_levels] for reference
    """

    # --- Prepare weighting ---
    if use_weights:
        lat_radians = np.deg2rad(latitudes)
        lat_weights = np.cos(lat_radians)  # shape [lat]
        lat_weights = np.clip(lat_weights, 0, None)

        def weighted_mean(data: np.ndarray) -> np.ndarray:
            # data: [samples, level, lat, lon]
            # apply lat weights across latitude axis only
            w = lat_weights[None, None, :, None]  # broadcast to [1,1,lat,1]
            weighted_sum = np.nansum(data * w, axis=2)  # sum over lat
            sum_w = np.nansum(w, axis=2)  # sum weights over lat
            return np.nanmean(weighted_sum / sum_w, axis=(0, -1))  # mean over samples & lon

        def weighted_stdev(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
            w = lat_weights[None, None, :, None]
            mean_exp = mean[None, :, None, None]
            var_lat = np.nansum(w * (data - mean_exp)**2, axis=2) / np.nansum(w, axis=2)
            return np.sqrt(np.nanmean(var_lat, axis=(0, -1)))  # mean over samples & lon
    else:
        def weighted_mean(data: np.ndarray) -> np.ndarray:
            return np.nanmean(data, axis=(0, -2, -1))

        def weighted_stdev(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
            return np.nanstd(data, axis=(0, -2, -1))

    def nan_min(data: np.ndarray) -> np.ndarray:
        return np.nanmin(data, axis=(0, -2, -1))

    def nan_max(data: np.ndarray) -> np.ndarray:
        return np.nanmax(data, axis=(0, -2, -1))

    stat_funcs = {
        "mean": weighted_mean,
        "stdev": weighted_stdev,
        "min": nan_min,
        "max": nan_max,
    }

    results: Dict[str, Dict[str, np.ndarray]] = {}

    for var_name, arr in store.items():
        if var_name not in PROFILE_CONFIG:
            continue
        if "level" not in arr.dims:
            continue

        lt_hours_all = None
        if "lead_time" in arr.coords:
            lt_hours_all = np.array(arr["lead_time"].values, dtype="timedelta64[h]").astype(int)

        # --- MODEL: per leadtime ---
        if selected_leadtimes is not None and lt_hours_all is not None:
            results[var_name] = {}
            stat_per_lt = {stat: [] for stat in summary_stats}
            leadtimes_sorted = [lt for lt in selected_leadtimes if np.any(lt_hours_all == lt)]

            for lt in leadtimes_sorted:
                idx_mask = (lt_hours_all == lt)
                subset = arr.isel(idx=np.where(idx_mask)[0]).values  # [n_samples, level, lat, lon]
                if subset.size == 0:
                    continue

                mean_val = weighted_mean(subset)
                for stat in summary_stats:
                    if stat == "stdev":
                        val = weighted_stdev(subset, mean_val)
                    elif stat == "mean":
                        val = mean_val
                    elif stat == "min":
                        val = nan_min(subset)
                    elif stat == "max":
                        val = nan_max(subset)
                    else:
                        continue
                    stat_per_lt[stat].append(val)

            # Stack per-leadtime → shape (num_leadtimes, n_levels)
            for stat in summary_stats:
                if stat_per_lt[stat]:
                    results[var_name][stat] = np.stack(stat_per_lt[stat], axis=0)

        # --- REFERENCE: collapse all leadtimes ---
        else:
            results[var_name] = {}
            subset = arr.values  # [n_samples, level, lat, lon]
            if subset.size == 0:
                continue

            mean_val = weighted_mean(subset)
            for stat in summary_stats:
                if stat == "stdev":
                    val = weighted_stdev(subset, mean_val)
                elif stat == "mean":
                    val = mean_val
                elif stat == "min":
                    val = nan_min(subset)
                elif stat == "max":
                    val = nan_max(subset)
                else:
                    continue
                results[var_name][stat] = val  # shape [n_levels]

    return results


def plot_profiles(
    results_model: Dict[str, Dict[str, np.ndarray]],
    results_ref: Optional[Dict[str, Dict[str, np.ndarray]]],
    pressure_levels: np.ndarray,
    output_dir: Path,
    summary_stats: List[str],
    model_name: str = "Model",
    ref_name: str = "Reference",
    leadtimes: Optional[List[int]] = None
):
    """
    Plot vertical profiles for each variable and summary stat.

    Model: results_model[var][stat] shape [num_leadtimes, levels]
    Reference: results_ref[var][stat] shape [levels]

    Vertical levels are plotted with pressure decreasing upward (inverted y-axis).
    """
    sns.set_style("whitegrid")
    out_dir = output_dir / "vertical_profiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    for var_name in results_model.keys():
        for stat in summary_stats:
            plt.figure(figsize=(6, 8))

            data_model = results_model[var_name].get(stat)
            if data_model is None:
                continue

            if data_model.ndim == 2:
                n_levels = data_model.shape[1]
            else:
                continue

            palette = sns.color_palette("tab10", n_colors=len(leadtimes))

            for i, lt in enumerate(leadtimes):
                prof = data_model[i]

                cfg = PROFILE_CONFIG.get(var_name, {}).get(stat, {})
                xmin, xmax = cfg.get("xmin", None), cfg.get("xmax", None)

                # Mask values outside allowed range
                mask = ~np.isnan(prof)
                if xmin is not None:
                    mask &= prof >= xmin
                if xmax is not None:
                    mask &= prof <= xmax

                plt.plot(
                    prof[mask], pressure_levels[mask],
                    label=f"{model_name}: Lt {lt}",
                    color=palette[i % len(palette)],
                    linewidth=1.8,
                )

            if results_ref and var_name in results_ref and stat in results_ref[var_name]:
                data_ref = results_ref[var_name][stat]
                mask_ref = ~np.isnan(data_ref)
                if xmin is not None:
                    mask_ref &= data_ref >= xmin
                if xmax is not None:
                    mask_ref &= data_ref <= xmax
                plt.plot(
                    data_ref[mask_ref], pressure_levels[mask_ref],
                    label=f"{ref_name}",
                    color="black",
                    linewidth=2.5,
                    linestyle="--",
                )

            ax = plt.gca()
            ax.invert_yaxis()
            plt.xlabel(f"{var_name.replace('_', ' ').capitalize()} ({stat})", fontsize=14)
            plt.ylim([1000, 5])
            plt.ylabel("Pressure Level", fontsize=14)
            plt.title(f"Vertical Profile of {var_name.replace('_', ' ').capitalize()} ({stat})", fontsize=14, weight='bold')

            # Use per-stat config if available
            cfg = None
            if var_name in PROFILE_CONFIG:
                cfg = PROFILE_CONFIG[var_name].get(stat) or None

            if cfg:
                if "xmin" in cfg and "xmax" in cfg:
                    plt.xlim(cfg["xmin"], cfg["xmax"])
                if "scale" in cfg:
                    ax.set_xscale(cfg["scale"])
            else:
                # Optional fallback scale for some stats
                if stat in ["min", "max"]:
                    ax.set_xscale("linear")

            ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6, width=1.2)
            ax.tick_params(axis='both', which='minor', direction='in', length=3, width=1)
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=1)

            leg = plt.legend(frameon=True, fontsize=12, loc='best', edgecolor='black', fancybox=True)
            leg.get_frame().set_alpha(0.9)

            plt.tight_layout()
            filename = f"{var_name}_{stat}.png"
            plt.savefig(out_dir / filename, dpi=300)
            print(f"Saved: {out_dir}/{filename}")
            plt.close()

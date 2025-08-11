import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns

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
        "mean": {"xmin": 1e-1, "xmax": 1e1, "scale": "log"},
        "stdev": {"xmin": 1e-1, "xmax": 1e1, "scale": "log"},
        "max": {"xmin": 1e-1, "xmax": 1e3, "scale": "log"},
        "min": {"xmin": 1e-4, "xmax": 1e1, "scale": "log"},
    },
    # Add more variables here with per-stat configs as needed
}

use_weights = True  # Set to False to disable latitude weighting


def compute_summary_stats(
    store: Dict[Tuple[str, np.datetime64, int], np.ndarray],
    latitudes: np.ndarray,
    selected_leadtimes: Optional[List[int]] = None,
    summary_stats: List[str] = ["mean", "stdev", "min", "max"],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute weighted or unweighted summary statistics over lat/lon per vertical level, sample-wise then averaged.
    Latitude weighting uses w = cos(lat) in radians if use_weights=True.

    Parameters:
    - store: dict of {(var_name, base_time, lead_time): np.ndarray}, shape [1, level, lat, lon]
    - latitudes: 1D array of latitudes corresponding to lat dimension, in degrees
    - selected_leadtimes: list of lead times for model, None for reference (all leadtimes)
    - summary_stats: list of stats to compute: mean, stdev, min, max

    Returns:
    - results: dict[var_name][stat] = np.ndarray shape [num_levels] for reference or
      shape [num_leadtimes, num_levels] for model.
    """

    if use_weights:
        # Convert latitudes to radians and compute weights shape [lat]
        lat_radians = np.deg2rad(latitudes)
        weights_1d = np.cos(lat_radians)
        weights_1d = np.clip(weights_1d, 0, None)
        weights_1d = weights_1d / np.sum(weights_1d)

        def weighted_mean(data: np.ndarray) -> np.ndarray:
            w2d = weights_1d[None, :, None]
            weighted_sum = np.nansum(data * w2d, axis=(-2, -1))
            total_weight = np.sum(weights_1d) * data.shape[-1]
            return weighted_sum / total_weight

        def weighted_stdev(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
            w2d = weights_1d[None, :, None]
            diff_sq = (data - mean[:, None, None]) ** 2
            weighted_var = np.nansum(diff_sq * w2d, axis=(-2, -1)) / (np.sum(weights_1d) * data.shape[-1])
            return np.sqrt(weighted_var)

        def nan_min(data: np.ndarray) -> np.ndarray:
            return np.nanmin(data, axis=(-2, -1))

        def nan_max(data: np.ndarray) -> np.ndarray:
            return np.nanmax(data, axis=(-2, -1))

        stat_funcs = {
            "mean": weighted_mean,
            "stdev": weighted_stdev,
            "min": nan_min,
            "max": nan_max,
        }

    else:
        # No weights: simple unweighted stats over lat/lon
        def simple_mean(data: np.ndarray) -> np.ndarray:
            return np.nanmean(data, axis=(-2, -1))

        def simple_stdev(data: np.ndarray, mean: np.ndarray) -> np.ndarray:
            return np.nanstd(data, axis=(-2, -1))

        def nan_min(data: np.ndarray) -> np.ndarray:
            return np.nanmin(data, axis=(-2, -1))

        def nan_max(data: np.ndarray) -> np.ndarray:
            return np.nanmax(data, axis=(-2, -1))

        stat_funcs = {
            "mean": simple_mean,
            "stdev": simple_stdev,
            "min": nan_min,
            "max": nan_max,
        }

    if selected_leadtimes is not None:
        data_accum = {}
        for (var_name, base_time, lead_time), arr in store.items():
            if lead_time not in selected_leadtimes:
                continue
            if var_name not in data_accum:
                data_accum[var_name] = {}
            if lead_time not in data_accum[var_name]:
                data_accum[var_name][lead_time] = {stat: [] for stat in summary_stats}

            data = np.array(arr).squeeze()
            if data.ndim != 3:
                continue

            for stat in summary_stats:
                if stat == "stdev":
                    mean_val = stat_funcs["mean"](data)
                    val = stat_funcs["stdev"](data, mean_val)
                else:
                    val = stat_funcs[stat](data)
                data_accum[var_name][lead_time][stat].append(val)

        results = {}
        for var_name, lt_dict in data_accum.items():
            results[var_name] = {}
            for stat in summary_stats:
                stat_per_lt = []
                leadtimes_sorted = sorted(lt_dict.keys())
                for lt in leadtimes_sorted:
                    arrs = lt_dict[lt][stat]
                    mean_arr = np.nanmean(arrs, axis=0)
                    stat_per_lt.append(mean_arr)
                results[var_name][stat] = np.stack(stat_per_lt, axis=0)

    else:
        data_accum = {}
        for (var_name, base_time, lead_time), arr in store.items():
            if var_name not in data_accum:
                data_accum[var_name] = {stat: [] for stat in summary_stats}

            data = np.array(arr).squeeze()
            if data.ndim != 3:
                continue

            for stat in summary_stats:
                if stat == "stdev":
                    mean_val = stat_funcs["mean"](data)
                    val = stat_funcs["stdev"](data, mean_val)
                else:
                    val = stat_funcs[stat](data)
                data_accum[var_name][stat].append(val)

        results = {}
        for var_name, stat_dict in data_accum.items():
            results[var_name] = {}
            for stat in summary_stats:
                arrs = stat_dict[stat]
                if len(arrs) == 0:
                    print(f"No valid samples for {var_name} stat {stat}")
                    continue
                stacked = np.stack(arrs, axis=0)
                results[var_name][stat] = np.nanmean(stacked, axis=0)

    return results


# plot_profiles remains unchanged
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
                plt.plot(
                    data_model[i], pressure_levels,
                    label=f"{model_name}: Lt {lt}",
                    color=palette[i % len(palette)],
                    linewidth=1.8,
                )

            if results_ref and var_name in results_ref and stat in results_ref[var_name]:
                data_ref = results_ref[var_name][stat]
                if data_ref.shape[0] == n_levels:
                    plt.plot(
                        data_ref, pressure_levels,
                        label=f"{ref_name}",
                        color='black',
                        linewidth=2.5,
                        linestyle='--',
                    )
                else:
                    print(f"Reference data shape mismatch for {var_name} {stat}: {data_ref.shape} vs {n_levels}")

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

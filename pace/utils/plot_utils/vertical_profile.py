import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns


def compute_summary_stats(
    store: Dict[Tuple[str, np.datetime64, int], np.ndarray],
    selected_leadtimes: Optional[List[int]] = None,
    summary_stats: List[str] = ["mean", "stdev", "min", "max"],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Compute summary statistics over lat/lon per vertical level, sample-wise then averaged.

    Parameters:
    - store: dict of {(var_name, base_time, lead_time): np.ndarray}, shape [1, level, lat, lon]
    - selected_leadtimes: list of lead times for model, None for reference (all leadtimes)
    - summary_stats: list of stats to compute: mean, stdev, min, max

    Returns:
    - results: dict[var_name][stat] = np.ndarray shape [num_levels] for reference or
      shape [num_leadtimes, num_levels] for model.
    """

    # Use nan-safe numpy funcs
    stat_funcs = {
        "mean": np.nanmean,
        "stdev": np.nanstd,
        "min": np.nanmin,
        "max": np.nanmax,
    }

    # Accumulate per sample statistics in lists:
    # For model: var_name -> lead_time -> stat -> list of arrays shape [levels]
    # For reference: var_name -> stat -> list of arrays shape [levels]
    if selected_leadtimes is not None:
        # MODEL
        data_accum = {}
        for (var_name, base_time, lead_time), arr in store.items():
            if lead_time not in selected_leadtimes:
                continue
            if var_name not in data_accum:
                data_accum[var_name] = {}
            if lead_time not in data_accum[var_name]:
                data_accum[var_name][lead_time] = {stat: [] for stat in summary_stats}

            data = np.array(arr)  # shape [1, level, lat, lon]
            data = data.squeeze()  # [level, lat, lon]
            if data.ndim != 3:
                # print(f"Skipping sample {var_name} {base_time} {lead_time}, unexpected shape {data.shape}")
                continue

            for stat in summary_stats:
                val = stat_funcs[stat](data, axis=(-2, -1))  # nan-aware over lat/lon
                data_accum[var_name][lead_time][stat].append(val)

        # Now average all per-sample stats for each var, lead_time, stat
        results = {}
        for var_name, lt_dict in data_accum.items():
            results[var_name] = {}
            for stat in summary_stats:
                # collect all lead_time arrays for this stat: shape [num_leadtimes, levels]
                stat_per_lt = []
                leadtimes_sorted = sorted(lt_dict.keys())
                for lt in leadtimes_sorted:
                    arrs = lt_dict[lt][stat]  # list of arrays shape [levels]
                    # mean over all samples for this leadtime
                    mean_arr = np.nanmean(arrs, axis=0)
                    stat_per_lt.append(mean_arr)
                results[var_name][stat] = np.stack(stat_per_lt, axis=0)  # shape [num_leadtimes, levels]
    
    else:
        # REFERENCE: accumulate stats over all samples (all base_time and lead_time)
        data_accum = {}
        for (var_name, base_time, lead_time), arr in store.items():
            if var_name not in data_accum:
                data_accum[var_name] = {stat: [] for stat in summary_stats}

            data = np.array(arr)  # [1, level, lat, lon]
            data = data.squeeze()  # [level, lat, lon]
            if data.ndim != 3:
                # print(f"Skipping sample {var_name} {base_time} {lead_time}, unexpected shape {data.shape}")
                continue

            for stat in summary_stats:
                val = stat_funcs[stat](data, axis=(-2, -1))
                data_accum[var_name][stat].append(val)

        results = {}
        for var_name, stat_dict in data_accum.items():
            results[var_name] = {}
            for stat in summary_stats:
                arrs = stat_dict[stat]  # list of arrays shape [levels,]
                if len(arrs) == 0:
                    print(f"No valid samples for {var_name} stat {stat}")
                    continue
                stacked = np.stack(arrs, axis=0)  # shape [num_samples, levels]
                # Average over all samples (axis=0) -> result shape [levels,]
                results[var_name][stat] = np.nanmean(stacked, axis=0)

    return results


def plot_profiles(
    results_model: Dict[str, Dict[str, np.ndarray]],
    results_ref: Optional[Dict[str, Dict[str, np.ndarray]]],
    output_dir: Path,
    summary_stats: List[str],
    model_name: str = "Model",
    ref_name: str = "Reference",
):
    """
    Plot vertical profiles for each variable and summary stat.

    Model: results_model[var][stat] shape [num_leadtimes, levels]
    Reference: results_ref[var][stat] shape [levels]

    vertical_levels inferred as integers [0,...,n_levels-1]
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

            # Determine number of levels from data_model shape
            if data_model.ndim == 2:
                n_levels = data_model.shape[1]
            else:  
                continue

            vertical_levels = np.arange(n_levels)

            # Plot model lead times
            num_leadtimes = data_model.shape[0]

            # Create color palette for num_leadtimes
            palette = sns.color_palette("tab10", n_colors=num_leadtimes)

            for i in range(num_leadtimes):
                plt.plot(
                    data_model[i], vertical_levels,
                    label=f"{model_name} Lt {i+1}",
                    color=palette[i % len(palette)],
                    linewidth=1.8,
                )

            # Plot reference as a thick black line if available
            if results_ref and var_name in results_ref and stat in results_ref[var_name]:
                data_ref = results_ref[var_name][stat]
                if data_ref.shape[0] == n_levels:
                    plt.plot(
                        data_ref, vertical_levels,
                        label=f"{ref_name}",
                        color='black',
                        linewidth=2.5,
                        linestyle='--',
                    )
                else:
                    print(f"Reference data shape mismatch for {var_name} {stat}: {data_ref.shape} vs {n_levels}")

            ax = plt.gca()
            ax.invert_yaxis()  # pressure or height: usually top-down
            plt.xlabel(f"{var_name.replace('_', ' ').capitalize()} ({stat})", fontsize=14)
            plt.ylabel("Vertical Level", fontsize=14)
            plt.title(f"Vertical Profile of {var_name.replace('_', ' ').capitalize()} ({stat})", fontsize=14, weight='bold')

            # Styling from histograms:
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

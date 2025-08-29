import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
from typing import Dict, List, Optional

# --- CONFIGURATION ---
TIME_SERIES_VARS = ["total_mass", "total_energy"]

def compute_valid_time(ds: xr.Dataset) -> xr.DataArray:
    """Compute valid_time = base_time + lead_time."""
    if "lead_time" not in ds.coords:
        raise ValueError("Dataset must have 'lead_time' coordinate")
    
    valid_time = ds["base_time"] + ds["lead_time"]
    return valid_time


def prepare_time_series(
    store: Dict[str, xr.DataArray],
    selected_leadtimes: Optional[List[int]] = None
) -> Dict[str, Dict[int, Dict[np.datetime64, float]]]:
    """
    Prepare time series for variables.

    Returns
    -------
    dict
        {var_name: {lead_time: {valid_time: aggregated_value}}}
    """
    ts_data: Dict[str, Dict[int, Dict[np.datetime64, float]]] = {}

    for var_name, arr in store.items():
        if var_name not in TIME_SERIES_VARS:
            continue

        lt_hours_all = arr["lead_time"].values.astype("timedelta64[h]").astype(int)
        valid_times = arr["base_time"].values + arr["lead_time"].values.astype("timedelta64[h]")

        # Select only requested lead times
        if selected_leadtimes is None:
            leadtimes_to_use = np.unique(lt_hours_all)
        else:
            leadtimes_to_use = [lt for lt in selected_leadtimes if np.any(lt_hours_all == lt)]

        if var_name not in ts_data:
            ts_data[var_name] = {}

        for lt in leadtimes_to_use:
            mask = lt_hours_all == lt
            sub_times = valid_times[mask]
            sub_values = arr.isel(idx=mask).values.squeeze(axis=-1)

            if sub_values.ndim > 1:
                sub_values = sub_values.reshape(sub_values.shape[0], -1).sum(axis=1)

            ts_data[var_name][lt] = dict(sorted(zip(sub_times, sub_values)))

    return ts_data


def aggregate_reference(
    ref_ts: Dict[str, Dict[int, Dict[np.datetime64, float]]]
) -> Dict[str, Dict[np.datetime64, float]]:
    """Aggregate reference across lead times into a single time series (mean over lead times)."""
    agg_ref: Dict[str, Dict[np.datetime64, float]] = {}

    for var_name, lt_dict in ref_ts.items():
        combined: Dict[np.datetime64, List[float]] = {}
        for lt, series in lt_dict.items():
            for t, v in series.items():
                combined.setdefault(t, []).append(v)

        agg_ref[var_name] = {t: float(np.mean(vals)) for t, vals in combined.items()}
        agg_ref[var_name] = dict(sorted(agg_ref[var_name].items()))

    return agg_ref


def plot_time_series(
    model_ts: Dict[str, Dict[int, Dict[np.datetime64, float]]],
    ref_ts: Optional[Dict[str, Dict[int, Dict[np.datetime64, float]]]],
    output_dir: Path,
    model_name: str = "Model",
    ref_name: str = "Reference"
):
    """Plot and save time series with multiple lead times for model, single aggregate for reference."""
    sns.set_style("whitegrid")
    out_dir = output_dir / "time_series"
    out_dir.mkdir(parents=True, exist_ok=True)

    palette = sns.color_palette("tab10", n_colors=max(len(lt_dict) for lt_dict in model_ts.values()))

    for var_name, lt_dict in model_ts.items():
        plt.figure(figsize=(10, 5))

        # Model: plot each lead time separately
        for i, (lt, series) in enumerate(sorted(lt_dict.items(), key=lambda x: x[0])):
            times, values = zip(*series.items())
            plt.plot(
                times, values,
                label=f"{model_name}: Lt {lt}h",
                linewidth=2,
                color=palette[i % len(palette)],
            )

        # Reference: aggregate into a single line
        if ref_ts and var_name in ref_ts:
            agg_ref = aggregate_reference(ref_ts)[var_name]
            
            times_ref, values_ref = zip(*agg_ref.items())
            plt.plot(
                times_ref, values_ref,
                label=ref_name,
                linewidth=2.5,
                linestyle="--",
                color="black",
            )

        plt.title(f"{var_name.replace('_', ' ').capitalize()} over Time", fontsize=16, weight="bold")
        plt.xlabel("Valid Time", fontsize=14)
        plt.ylabel(var_name.replace('_', ' ').capitalize(), fontsize=14)
        plt.xticks(rotation=30)
        plt.legend(frameon=True, fontsize=12, loc="best", edgecolor="black", fancybox=True)
        plt.tight_layout()
        plt.savefig(out_dir / f"{var_name}.png", dpi=300)
        print(f"Saved: {out_dir}/{var_name}.png")
        plt.close()

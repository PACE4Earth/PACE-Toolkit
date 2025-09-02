import sys
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import UnifiedDataset

# -------------------------------
# Config: choose model member handling
# -------------------------------
USE_MEAN_MEMBER = False   # if True → mean over 8 members, else use SINGLE_MEMBER_IDX
SINGLE_MEMBER_IDX = 0    # which member to use when USE_MEAN_MEMBER = False

# -------------------------------
# Config: plot ranges and colorbar limits
# -------------------------------
XLIM = (3, 15)   # e.g., (5, 15) for longitude, or None for auto
YLIM = (47, 54)   # e.g., (45, 55) for latitude, or None for auto

# Will be filled dynamically based on model dataset (so reference shares same limits)
COLORBAR_LIMITS = {
    "2m_temperature": None,
    "total_precipitation": (0, 50),
    "vmax_10m": (0, 25),
}


def process_var(var, var_name=None):
    """Process variable: handle shapes (1, 8, H, W) and (H, W), convert units if needed."""
    if hasattr(var, 'cpu'):
        var = var.cpu().numpy()
    
    if var.ndim == 4:  # Model: (1, 8, H, W)
        if USE_MEAN_MEMBER:
            var = var[0].mean(axis=0)  # mean over second dim
        else:
            var = var[0, SINGLE_MEMBER_IDX]  # select one member
    elif var.ndim == 2:  # Reference: (H, W)
        pass  # already 2D

    # Convert units
    units = ""
    if var_name == "2m_temperature":
        var = var - 273.15  # K -> °C
        units = "°C"
    elif var_name == "total_precipitation":
        units = "mm/6h"
    elif var_name == "vmax_10m":
        units = "m/s"
    return var, units


def plot_sample(sample, name, out_dir, lat, lon, lon_ticks, lat_ticks, time_str, is_model=True, lead_time=None):
    """Plot a sample with 4-panel layout and save to file."""
    temp, temp_units = process_var(sample["2m_temperature"], "2m_temperature")
    precip, precip_units = process_var(sample["total_precipitation"], "total_precipitation")
    vmax, vmax_units = process_var(sample["vmax_10m"], "vmax_10m")

    fig, axes = plt.subplots(4, 1, figsize=(6, 16), constrained_layout=True)  # 1 column, 4 rows
    if is_model:
        plt.suptitle(f"{name} | Base: {time_str} | Lead: {lead_time}h", fontsize=16)
    else:
        plt.suptitle(f"{name} | Valid: {time_str}", fontsize=16)

    cmap_temp = 'coolwarm'
    cmap_precip = 'Blues'
    cmap_vmax = 'viridis'
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    def plot_var(ax, data, cmap, title, units, vmin=None, vmax=None):
        im = ax.imshow(data, cmap=cmap, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} [{units}]")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels([f"{v:.0f}°" for v in lon_ticks])
        ax.set_yticklabels([f"{v:.0f}°" for v in lat_ticks])
        if XLIM: ax.set_xlim(XLIM)
        if YLIM: ax.set_ylim(YLIM)
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(YLIM))))
        plt.colorbar(im, ax=ax, fraction=0.7)
        return im

    # 1) Temperature
    plot_var(axes[0], temp, cmap_temp, "2m Temperature", temp_units,
             *COLORBAR_LIMITS["2m_temperature"] if COLORBAR_LIMITS["2m_temperature"] else (None, None))

    # 2) Precipitation
    plot_var(axes[1], precip, cmap_precip, "Total Precipitation", precip_units,
             *COLORBAR_LIMITS["total_precipitation"] if COLORBAR_LIMITS["total_precipitation"] else (None, None))

    # 3) Vmax
    plot_var(axes[2], vmax, cmap_vmax, "Vmax 10m", vmax_units,
             *COLORBAR_LIMITS["vmax_10m"] if COLORBAR_LIMITS["vmax_10m"] else (None, None))

    # 4) Overlay
    axes[3].imshow(temp, cmap=cmap_temp, extent=extent, origin='lower',
                   vmin=COLORBAR_LIMITS["2m_temperature"][0] if COLORBAR_LIMITS["2m_temperature"] else None,
                   vmax=COLORBAR_LIMITS["2m_temperature"][1] if COLORBAR_LIMITS["2m_temperature"] else None)
    cs_precip = axes[3].contour(precip, colors='blue', levels=[5, 10, 15], alpha=0.7, extent=extent)
    axes[3].clabel(cs_precip, inline=1, fontsize=8)
    axes[3].contourf(vmax, levels=[10, 15, 20, 25, 30, 40, 50],
                     colors='none', hatches=['/', '//', '///', '////', '/////', '//////'],
                     alpha=0.1, extent=extent)
    axes[3].set_title("Overlay: Temp + Precip + Vmax")
    axes[3].set_xlabel("Longitude")
    axes[3].set_ylabel("Latitude")
    axes[3].set_xticks(lon_ticks)
    axes[3].set_yticks(lat_ticks)
    axes[3].set_xticklabels([f"{v:.0f}°" for v in lon_ticks])
    axes[3].set_yticklabels([f"{v:.0f}°" for v in lat_ticks])
    if XLIM: axes[3].set_xlim(XLIM)
    if YLIM: axes[3].set_ylim(YLIM)

    # plt.tight_layout()
    # pos = axes[3].get_position()
    # axes[3].set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])

    if is_model:
        filename = f"{name}_{time_str}_{lead_time}h.png"
    else:
        filename = f"{name}_{time_str}.png"

    plots_dir = out_dir / filename
    plt.savefig(plots_dir, dpi=300)
    plt.close(fig)
    print(f"Saved to: {plots_dir}")


def plot_difference(truth, prediction, name, out_dir, lat, lon, lon_ticks, lat_ticks, time_str, lead_time):
    """Plot truth vs prediction vs difference (3x3 layout)."""
    t_truth, t_units = process_var(truth["2m_temperature"], "2m_temperature")
    p_truth, p_units = process_var(truth["total_precipitation"], "total_precipitation")
    v_truth, v_units = process_var(truth["vmax_10m"], "vmax_10m")

    t_pred, _ = process_var(prediction["2m_temperature"], "2m_temperature")
    p_pred, _ = process_var(prediction["total_precipitation"], "total_precipitation")
    v_pred, _ = process_var(prediction["vmax_10m"], "vmax_10m")

    # Differences
    t_diff = t_pred - t_truth
    v_diff = v_pred - v_truth
    p_diff = p_pred - p_truth

    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    plt.suptitle(f"Base: {time_str} | Lead: {lead_time}h", fontsize=18)

    cmap_temp = "coolwarm"
    cmap_precip = "Blues"
    cmap_vmax = "viridis"
    cmap_diff = "RdBu_r"
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    def plot_var(ax, data, cmap, title, units, diff=False, vmin=None, vmax=None):
        if diff:
            vmax = np.nanmax(np.abs(data))
            vmin, vmax = -vmax, vmax  # symmetric around 0
            im = ax.imshow(data, cmap=cmap, extent=extent, origin="lower", vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, extent=extent, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"{title} [{units}]", fontsize=14)
        ax.set_xticks(lon_ticks)
        ax.set_yticks(lat_ticks)
        ax.set_xticklabels([f"{v:.0f}°" for v in lon_ticks])
        ax.set_yticklabels([f"{v:.0f}°" for v in lat_ticks])
        if XLIM: ax.set_xlim(XLIM)
        if YLIM: ax.set_ylim(YLIM)
        ax.set_aspect(1 / np.cos(np.deg2rad(np.mean(YLIM))))
        plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02, fraction=0.7)
        return im

    # Row 1: Temp
    plot_var(axes[0, 0], t_truth, cmap_temp, "COSMO-REA2 T2m", t_units,
             vmin=COLORBAR_LIMITS["2m_temperature"][0] if COLORBAR_LIMITS["2m_temperature"] else None,
             vmax=COLORBAR_LIMITS["2m_temperature"][1] if COLORBAR_LIMITS["2m_temperature"] else None)
    plot_var(axes[0, 1], t_pred, cmap_temp, f"CORRDIFF member {SINGLE_MEMBER_IDX+1} T2m", t_units,
             vmin=COLORBAR_LIMITS["2m_temperature"][0] if COLORBAR_LIMITS["2m_temperature"] else None,
             vmax=COLORBAR_LIMITS["2m_temperature"][1] if COLORBAR_LIMITS["2m_temperature"] else None)
    plot_var(axes[0, 2], t_diff, cmap_diff, "Pred. - Truth", t_units, diff=True)

    # Row 2: Precip
    plot_var(axes[1, 0], p_truth, cmap_precip, "COSMO-REA2 Precip", p_units,
             vmin=COLORBAR_LIMITS["total_precipitation"][0] if COLORBAR_LIMITS["total_precipitation"] else None,
             vmax=COLORBAR_LIMITS["total_precipitation"][1] if COLORBAR_LIMITS["total_precipitation"] else None)
    plot_var(axes[1, 1], p_pred, cmap_precip, f"CORRDIFF member {SINGLE_MEMBER_IDX+1} Precip", p_units,
             vmin=COLORBAR_LIMITS["total_precipitation"][0] if COLORBAR_LIMITS["total_precipitation"] else None,
             vmax=COLORBAR_LIMITS["total_precipitation"][1] if COLORBAR_LIMITS["total_precipitation"] else None)
    plot_var(axes[1, 2], p_diff, cmap_diff, "Pred. - Truth", p_units, diff=True)

    # Row 3: Vmax
    plot_var(axes[2, 0], v_truth, cmap_vmax, "COSMO-REA2 Vmax", v_units,
             vmin=COLORBAR_LIMITS["vmax_10m"][0] if COLORBAR_LIMITS["vmax_10m"] else None,
             vmax=COLORBAR_LIMITS["vmax_10m"][1] if COLORBAR_LIMITS["vmax_10m"] else None)
    plot_var(axes[2, 1], v_pred, cmap_vmax, f"CORRDIFF member {SINGLE_MEMBER_IDX+1} Vmax", v_units,
             vmin=COLORBAR_LIMITS["vmax_10m"][0] if COLORBAR_LIMITS["vmax_10m"] else None,
             vmax=COLORBAR_LIMITS["vmax_10m"][1] if COLORBAR_LIMITS["vmax_10m"] else None)
    plot_var(axes[2, 2], v_diff, cmap_diff, "Pred. - Truth", v_units, diff=True)

    # plt.tight_layout(rect=[0, 0, 1, 1])
    filename = f"{name}_diff_{time_str}_{lead_time}h.png"
    plots_dir = out_dir / filename
    plt.savefig(plots_dir, dpi=300)
    plt.close(fig)
    print(f"Saved diff plot to: {plots_dir}")


def main():
    start_time = time.perf_counter()
    config_path = Path(os.environ["CONFIG_PATH"])
    out_dir = Path(os.environ["PLOTS_DIR_PATH"]) / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dataset = UnifiedDataset(config_path, dataset_key="model")
    reference_dataset = UnifiedDataset(config_path, dataset_key="reference")
    lat = model_dataset.grid["lat"]
    lon = model_dataset.grid["lon"]

    # Define tick intervals
    lon_ticks = np.arange(lon.min(), lon.max(), 3)
    lat_ticks = np.arange(lat.min(), lat.max(), 3)

    # -------------------------------
    # Compute global colorbar limits from model dataset
    # -------------------------------
    global COLORBAR_LIMITS
    vars_to_check = ["2m_temperature"]
    mins, maxs = {v: [] for v in vars_to_check}, {v: [] for v in vars_to_check}
    for i in range(len(model_dataset)):
        s = model_dataset[i]
        for v in vars_to_check:
            arr, _ = process_var(s[v], v)
            mins[v].append(np.nanmin(arr))
            maxs[v].append(np.nanmax(arr))
    for v in vars_to_check:
        COLORBAR_LIMITS[v] = (np.nanmin(mins[v]), np.nanmax(maxs[v]))
    print("Colorbar limits:", COLORBAR_LIMITS)

    # -------------------------------
    # Loop over model samples
    # -------------------------------
    for i, (file_path, base_time, lead_idx, leadtimes, o) in enumerate(model_dataset.samples):
        sample = model_dataset[i]
        ref_sample = reference_dataset[i]  # align by index (assuming same order)
        base_time = sample['base_time']
        lead_time = sample['lead_time']
        base_str = base_time.strftime("%Y%m%d_%H")
        lead_hours = int(lead_time.total_seconds() // 3600)
        if not lead_hours == 6:
            continue
        name = "CORRDIFF"

        # Normal plot
        # plot_sample(sample, name, out_dir, lat, lon, lon_ticks, lat_ticks, base_str, is_model=True, lead_time=lead_hours)

        # Difference plot
        plot_difference(ref_sample, sample, name, out_dir, lat, lon, lon_ticks, lat_ticks, base_str, lead_hours)

    # -------------------------------
    # Single reference sample (first one)
    # -------------------------------
    ref_sample = reference_dataset[0]
    valid_time = reference_dataset.valid_times_for_samples[0]
    valid_str = valid_time.strftime("%Y%m%d_%H")
    name = "COSMO-REA2"

    # plot_sample(ref_sample, name, out_dir, lat, lon, lon_ticks, lat_ticks, valid_str, is_model=False)

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time}")


if __name__ == "__main__":
    main()

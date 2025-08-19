import sys
import os
import matplotlib.pyplot as plt
import time
import numpy as np
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import UnifiedDataset

def process_var(var, var_name=None):
    """Process variable: handle shapes (1, 8, H, W) and (H, W), convert units if needed."""
    if hasattr(var, 'cpu'):
        var = var.cpu().numpy()
    if var.ndim == 4:  # (1, 8, H, W)
        var = var[0].mean(axis=0)  # mean over second dim
    elif var.ndim == 3:  # (8, H, W) without singleton first dim
        var = var.mean(axis=0)

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


def main():
    start_time = time.perf_counter()
    config_path = "/p/project/hclimrep/vas1/PACE-Toolkit/pace/configs/config_corrdiff.json"
    out_dir = Path("/p/project/hclimrep/vas1/PACE-Toolkit/pace/plots/case_studies")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dataset = UnifiedDataset(config_path, dataset_key="model")
    lat = model_dataset.grid["lat"]
    lon = model_dataset.grid["lon"]

    # Define tick intervals
    lon_ticks = np.arange(lon.min(), lon.max(), 3)
    lat_ticks = np.arange(lat.min(), lat.max(), 3)

    for i, (file_path, base_time, lead_idx, leadtimes, o) in enumerate(model_dataset.samples):
        valid_time = model_dataset.valid_times_for_samples[i]
        print(f"Base: {base_time}, LeadIdx: {lead_idx}, Valid: {valid_time}, File: {file_path.name}")
        sample = model_dataset[i]

        # -------------------------------
        # 2x2 panel plot
        # -------------------------------
        temp, temp_units = process_var(sample["2m_temperature"], "2m_temperature")
        precip, precip_units = process_var(sample["total_precipitation"], "total_precipitation")
        vmax, vmax_units = process_var(sample["vmax_10m"], "vmax_10m")

        name = "CORRDIFF" if sample["2m_temperature"].ndim == 4 else "COSMO-REA2"
        base_time = sample['base_time']
        lead_time = sample['lead_time']
        base_str = base_time.strftime("%Y%m%d_%H")
        lead_hours = int(lead_time.total_seconds() // 3600)
        lead_str = f"{lead_hours}h"

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        plt.suptitle(f"{name} | Base: {base_str} | Lead: {lead_str}", fontsize=16)

        cmap_temp = 'coolwarm'
        cmap_precip = 'Blues'
        cmap_vmax = 'viridis'
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]

        # Function to plot image with ticks and colorbar
        def plot_var(ax, data, cmap, title, units):
            im = ax.imshow(data, cmap=cmap, extent=extent, origin='lower')
            ax.set_title(f"{title} [{units}]")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xticks(lon_ticks)
            ax.set_yticks(lat_ticks)
            ax.set_xticklabels([f"{v:.0f}°" for v in lon_ticks])
            ax.set_yticklabels([f"{v:.0f}°" for v in lat_ticks])
            cbar = plt.colorbar(im, ax=ax, fraction=0.05, shrink=0.75, pad=0.03)  # smaller fraction
            return im

        # 1) Temperature
        im0 = plot_var(axes[0, 0], temp, cmap_temp, "2m Temperature", temp_units)
        # 2) Precipitation
        im1 = plot_var(axes[0, 1], precip, cmap_precip, "Total Precipitation", precip_units)
        # 3) Vmax
        im2 = plot_var(axes[1, 0], vmax, cmap_vmax, "Vmax 10m", vmax_units)

        # 4) Overlay
        axes[1, 1].imshow(temp, cmap=cmap_temp, extent=extent, origin='lower')
        cs_precip = axes[1, 1].contour(precip, colors='blue', levels=[5, 10, 15], alpha=0.7, extent=extent)
        axes[1, 1].clabel(cs_precip, inline=1, fontsize=8)
        axes[1, 1].contourf(vmax, levels=[10, 15, 20, 25], colors='none', hatches=['/', '//', 'xx'], alpha=0.1, extent=extent)
        
        axes[1, 1].set_title("Overlay: Temp + Precip + Vmax")
        axes[1, 1].set_xlabel("Longitude")
        axes[1, 1].set_ylabel("Latitude")
        axes[1, 1].set_xticks(lon_ticks)
        axes[1, 1].set_yticks(lat_ticks)
        axes[1, 1].set_xticklabels([f"{v:.0f}°" for v in lon_ticks])
        axes[1, 1].set_yticklabels([f"{v:.0f}°" for v in lat_ticks])

        plt.tight_layout()
        plt.savefig(out_dir / f"{name}_{base_str}_{lead_str}.png", dpi=300)
        print(f"Saved to: {out_dir}/{name}_{base_str}_{lead_str}.png")
        # break

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time}")

if __name__ == "__main__":
    main()

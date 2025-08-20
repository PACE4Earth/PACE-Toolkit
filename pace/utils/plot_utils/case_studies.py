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
        sample = model_dataset[i]
        name = "CORRDIFF" if sample["2m_temperature"].ndim == 4 else "COSMO-REA2"
        base_time = sample['base_time']
        lead_time = sample['lead_time']
        valid_time = model_dataset.valid_times_for_samples[i]
        base_str = base_time.strftime("%Y%m%d_%H")
        lead_hours = int(lead_time.total_seconds() // 3600)
        lead_str = f"{lead_hours}h"
        valid_str = valid_time.strftime("%Y%m%d_%H")

        # -------------------------------
        # 4x1 panel plot
        # -------------------------------
        temp, temp_units = process_var(sample["2m_temperature"], "2m_temperature")
        precip, precip_units = process_var(sample["total_precipitation"], "total_precipitation")
        vmax, vmax_units = process_var(sample["vmax_10m"], "vmax_10m")

        fig, axes = plt.subplots(4, 1, figsize=(6, 16))  # 1 column, 4 rows
        if name == "CORRDIFF":
            plt.suptitle(f"{name} | Base: {base_str} | Lead: {lead_str}", fontsize=16)
        else: 
            plt.suptitle(f"{name} | Valid: {valid_str}", fontsize=16)

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
            cbar = plt.colorbar(im, ax=ax)  
            return im

        # 1) Temperature
        im0 = plot_var(axes[0], temp, cmap_temp, "2m Temperature", temp_units)

        # 2) Precipitation
        im1 = plot_var(axes[1], precip, cmap_precip, "Total Precipitation", precip_units)

        # 3) Vmax
        im2 = plot_var(axes[2], vmax, cmap_vmax, "Vmax 10m", vmax_units)

        # 4) Overlay
        axes[3].imshow(temp, cmap=cmap_temp, extent=extent, origin='lower')
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

        plt.tight_layout()  
        pos = axes[3].get_position()  
        axes[3].set_position([pos.x0 - 0.05, pos.y0, pos.width, pos.height])  

        if name == "CORRDIFF":
            plots_dir = out_dir / f"{name}_{base_str}_{lead_str}.png"
            plt.savefig(plots_dir, dpi=300)
        else: 
            plots_dir = out_dir / f"{name}_{valid_str}.png"
            plt.savefig(plots_dir, dpi=300)
        print(f"Saved to: {plots_dir}")

        if name == "COSMO-REA2":
            break

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time}")

if __name__ == "__main__":
    main()

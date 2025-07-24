import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

# Optional: Seaborn for prettier plots
import seaborn as sns
sns.set(style="whitegrid")

# CONFIGURATION
MODEL = "graphcast"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", MODEL)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METRICS = ["hydrostatic_balance_abs_error", "hydrostatic_balance_rel_error"]

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_all_outputs():
    """Load all available .nc files from output directory into a list of datasets."""
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".nc")])
    datasets = [xr.open_dataset(os.path.join(OUTPUT_DIR, f)) for f in files]
    
    # fix later
    for i in range(len(datasets)):
        datasets[i]["lead_time"] = datasets[i]["lead_time"] + 6

    print(datasets)
    return datasets

def plot_spatial_slice(ds, metric, lead_time, level, show_geopotential=True):
    """Plot spatial slice for given metric, lead_time, and level"""
    if metric not in ds:
        print(f"{metric} not found in dataset.")
        return

    data = ds[metric].sel(lead_time=lead_time, level=level, method="nearest")
    base_time_hour = ds['base_time'].values.astype('datetime64[h]')
    lon = ds['lon']
    lat = ds['lat'] 
    Lon, Lat = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(Lon, Lat, data, cmap='viridis', norm=colors.LogNorm(vmin=1e-3, vmax=np.max(data)))

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(f"{metric.replace('_', ' ').capitalize()}", fontsize=14)

    # Add geopotential contours if requested
    if show_geopotential and 'geopotential' in ds:
        geo = ds['geopotential'].sel(level=level, lead_time=lead_time, method="nearest")
        geopotential_height = geo / 9.80665  # convert from m²/s² to meters
        cs = ax.contour(lon, lat, geopotential_height, levels=20, colors='white', linewidths=0.9)
        ax.clabel(cs, inline=False, fontsize=8, fmt='%1.0f', use_clabeltext=True); [txt.set_bbox(dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')) for txt in cs.labelTexts]

    # if show_q and 'specific_humidity' in ds:
    #     geo = ds['specific_humidity'].sel(level=level, lead_time=lead_time, method="nearest")
    #     geopotential_height = geo / 9.80665  # convert from m²/s² to meters
    #     cs = ax.contour(lon, lat, geopotential_height, levels=20, colors='white', linewidths=0.9)
    #     ax.clabel(cs, inline=False, fontsize=8, fmt='%1.0f', use_clabeltext=True); [txt.set_bbox(dict(facecolor='black', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2')) for txt in cs.labelTexts]

    ax.set_title(f'Graphcast: {metric.replace('_', ' ').capitalize()}\nLevel: {level} hPa | Base: {base_time_hour} | Lead: 6h', fontsize=16)
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_xlim([min(lon), max(lon)])
    ax.set_ylim([min(lat), max(lat)])
    # ax.set_xlim([0, 100])
    # ax.set_ylim([20, 80])
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}°'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.0f}°'))

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{metric}_level{level}_lead{lead_time}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_vertical_leadtime_profile(ds, n_leads=5):
    """Plot vertical profile (lead_time vs level) averaged over lat/lon"""

    if metric not in ds:
        print(f"{metric} not found in dataset.")
        return
    if "lead_time" not in ds.dims:
        print("No lead_time dimension in dataset.")
        return
    
    mean_profile = ds[metric].mean(dim=["lat", "lon"])

    # Select n_leads evenly spaced lead times
    total_leads = ds.dims["lead_time"]
    ds["lead_time"] = ds["lead_time"]
    lead_indices = np.linspace(0, total_leads - 1, n_leads, dtype=int)
    profiles_to_plot = mean_profile.isel(lead_time=lead_indices)
    # x_min = np.nanpercentile(profiles_to_plot.values, 5)
    # x_max = np.nanpercentile(profiles_to_plot.values, 95)
    # print(x_min, x_max)

    plt.figure(figsize=(6, 8))
    ax = plt.gca()

    for idx in lead_indices:
        lead = ds["lead_time"].isel(lead_time=idx).item()
        ax.plot(mean_profile.isel(lead_time=idx), ds["level"], label=f"Graphcast: Lead time {lead}h", linewidth=2)

    base_time_hour = ds['base_time'].values.astype('datetime64[h]')

    ax.invert_yaxis()  # So level 1000 is at bottom, 100 at top
    ax.set_yticks([1000, 850, 700, 500, 300, 100, 1])
    ax.set_ylim(bottom=1000)

    # Style axes and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('black')

    ax.tick_params(axis='both', colors='black')
    ax.grid(True, which='both', color='black', linestyle='--', linewidth=0.5, alpha=0.4)

    ax.set_title(f"{metric.replace('_', ' ').capitalize()}\nMean over lat/lon\nBase Time: {base_time_hour}", fontsize=16)
    ax.set_xlabel(f"{metric.replace('_', ' ')} mean", fontsize=14)
    ax.set_ylabel("Pressure Level (hPa)", fontsize=14)
    # ax.set_xlim(x_min, x_max)
    ax.set_xscale("log")
    ax.legend(frameon=True, loc='center right', fontsize=12,
              framealpha=0.9, edgecolor='black', fancybox=True)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f"{metric}_vertical_leadtime_profile.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()



def compute_global_skill_stats(ds, metric):
    """Compute mean and std of the metric for summary."""
    if metric not in ds:
        return None
    arr = ds[metric].values
    return {
        "mean": np.nanmean(arr),
        "std": np.nanstd(arr),
        "min": np.nanmin(arr),
        "max": np.nanmax(arr),
    }

def process_metric(metric, datasets):
    """Full workflow for one metric."""
    print(f"\nProcessing metric: {metric}")
    for i, ds in enumerate(datasets):
        if metric not in ds:
            continue
        plot_spatial_slice(ds, metric, lead_time=0, level=500)
        # plot_vertical_leadtime_profile(ds)

if __name__ == "__main__":
    datasets = load_all_outputs()
    for metric in METRICS:
        process_metric(metric, datasets)
        
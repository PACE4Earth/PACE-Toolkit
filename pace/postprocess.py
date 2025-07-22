import os
import numpy as np
import xarray as xr
import matplotlib.colors as colors
import matplotlib.pyplot as plt

# Optional: Seaborn for prettier plots
import seaborn as sns
sns.set(style="whitegrid")

# CONFIGURATION
MODEL = "graphcast"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", MODEL)
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METRICS = ["geostrophic_balance"]  # add your metrics

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_all_outputs():
    """Load all available .nc files from output directory into a list of datasets."""
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(".nc")])
    datasets = [xr.open_dataset(os.path.join(OUTPUT_DIR, f)) for f in files]
    print(datasets)
    return datasets

def plot_metric_level_slice(ds, metric, level_index, show_geopotential=True):
    """Plot a single level slice for a given metric."""
    if metric not in ds:
        print(f"Metric {metric} not found in dataset.")
        return

    data = ds[metric].isel(level=level_index)
    base_time_hour = ds['base_time'].values.astype('datetime64[h]')
    lon = ds['lon']
    lat = ds['lat']
    Lon, Lat = np.meshgrid(lon, lat)

    plt.figure(figsize=(12, 5))
    plt.pcolormesh(Lon, Lat, data, cmap='Grays', norm=colors.LogNorm(vmin=1e-2, vmax=10), shading='auto')
    plt.colorbar(label='Ageostrophic / Geostrophic Wind Magnitude', )

     # Add geopotential contours if requested and available
    if show_geopotential and 'geopotential' in ds:
        geo = ds['geopotential'].isel(level=level_index)
        geopotential_height = geo / 9.80665  # in meters
        cs = plt.contour(lon, lat, geopotential_height, colors='white', linewidths=0.8)
        plt.clabel(cs, inline=True, fontsize=8, fmt='%1.0f')

    plt.title(f'Ratio of Ageostrophic to Geostrophic Wind, Level: {level_index}, Time: {base_time_hour}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)
    plt.xlim([min(lon), max(lon)])
    plt.ylim([min(lat), max(lat)])
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{metric}_level{level_index}.png"), dpi=300)
    plt.close()

def plot_mean_geostrophic_wind_profile(ds, n_leads=5):
    """Calculate and plot the vertical profile of mean geostrophic wind per pressure level."""

    metric = "geostrophic_balance"
    if metric not in ds:
        print(f"{metric} not found in dataset.")
        return
    if "lead_time" not in ds.dims:
        print("No lead_time dimension in dataset.")
        return

    # Select 5 evenly spaced lead times
    total_leads = ds.dims["lead_time"]
    lead_indices = np.linspace(0, total_leads - 1, n_leads, dtype=int)

    plt.figure(figsize=(6, 8))
    for idx in lead_indices:
        lead = ds["lead_time"].isel(lead_time=idx).item()
        profile = ds[metric].isel(lead_time=idx).mean(dim=["lat", "lon"])
        plt.plot(profile, ds["level"], label=f"Lead time: {lead}h")

    base_time_hour = ds['base_time'].values.astype('datetime64[h]')

    plt.gca().invert_yaxis()  # So level 0 (top) is at top, 36 bottom

    plt.title(f"Mean Ageo/Geostrophic Wind Ratio, Time: {base_time_hour}")
    plt.xlabel("Mean Ageo/Geostrophic Wind Ratio")
    plt.ylabel("Pressure Level Index (0=top)")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "mean_geostrophic_wind_vertical_profile.png"), dpi=300)
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
        plot_metric_level_slice(ds.isel(lead_time=0), metric, level_index=30)
        plot_mean_geostrophic_wind_profile(ds) 

if __name__ == "__main__":
    datasets = load_all_outputs()
    for metric in METRICS:
        process_metric(metric, datasets)
        
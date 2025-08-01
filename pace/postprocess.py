import os
import re
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import seaborn as sns
sns.set(style="whitegrid")

# CONFIGURATION
MODEL = "graphcast"
REFERENCE = "era5"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
METRICS = ["geostrophic_balance"]

FULLFIELDS_DIR = os.path.join(OUTPUT_DIR, "fullfields")
SUMMARY_DIR = os.path.join(OUTPUT_DIR, "summary")

FULLFIELDS_PLOT_DIR = os.path.join(PLOTS_DIR, "fullfields")
SUMMARY_PLOT_DIR = os.path.join(PLOTS_DIR, "summary")
os.makedirs(FULLFIELDS_PLOT_DIR, exist_ok=True)
os.makedirs(SUMMARY_PLOT_DIR, exist_ok=True)

def load_summary_outputs():
    model_dir = os.path.join(SUMMARY_DIR, MODEL)
    ref_dir = os.path.join(SUMMARY_DIR, REFERENCE)

    summary_model = None
    summary_ref = None

    model_files = sorted([f for f in os.listdir(model_dir) if f.endswith(".nc")])
    if model_files:
        summary_model = xr.open_dataset(os.path.join(model_dir, model_files[0]))

    ref_files = sorted([f for f in os.listdir(ref_dir) if f.endswith(".nc")])
    if ref_files:
        summary_ref = xr.open_dataset(os.path.join(ref_dir, ref_files[0]))

    print("\n[SUMMARY MODEL DATASET]\n", summary_model)
    if summary_ref:
        print("\n[SUMMARY REF DATASET]\n", summary_ref)

    return summary_model, summary_ref

def load_fullfields_outputs():
    model_dir = os.path.join(FULLFIELDS_DIR, MODEL)
    ref_dir = os.path.join(FULLFIELDS_DIR, REFERENCE)

    def parse_files(directory):
        data = []
        files = sorted([f for f in os.listdir(directory) if f.endswith(".nc")])
        for f in files:
            match = re.search(r"lead(\d+)", f)
            if match:
                lead_time = int(match.group(1))
                path = os.path.join(directory, f)
                ds = xr.open_dataset(path)
                data.append((lead_time, ds))
        return sorted(data, key=lambda x: x[0])

    model_data = parse_files(model_dir)
    ref_data = parse_files(ref_dir)

    if model_data:
        print("\n[FULLFIELDS MODEL FIRST DATASET]\n", model_data[0][1])
    if ref_data:
        print("\n[FULLFIELDS REF FIRST DATASET]\n", ref_data[0][1])

    return model_data, ref_data

def plot_spatial_slice(ds, metric, lead_time, level, title_prefix, plot_path):
    if metric not in ds:
        print(f"{metric} not found in dataset.")
        return

    data = ds[metric].sel(lead_time=lead_time, level=level, method="nearest")
    lon = ds['lon']
    lat = ds['lat'] 
    Lon, Lat = np.meshgrid(lon, lat)

    fig, ax = plt.subplots(figsize=(12, 5))
    pcm = ax.pcolormesh(Lon, Lat, data, cmap='viridis', norm=colors.LogNorm(vmin=1e-3, vmax=np.max(data)))
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(f"{metric.replace('_', ' ').capitalize()}", fontsize=14)

    ax.set_title(f"{title_prefix}: {metric} | Level: {level} hPa | Lead: {lead_time}h", fontsize=16)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

def plot_vertical_leadtime_profile(ds_model, ds_ref, metric):
    if metric not in ds_model:
        print(f"{metric} not found in model dataset.")
        return

    mean_profile = ds_model[metric]
    total_leads = ds_model.dims["lead_time"]
    lead_indices = np.linspace(0, total_leads - 1, 5, dtype=int)
    profiles_to_plot = mean_profile.isel(lead_time=lead_indices)

    plt.figure(figsize=(6, 8))
    ax = plt.gca()

    for idx in lead_indices:
        lead = ds_model["lead_time"].isel(lead_time=idx).item()
        ax.plot(mean_profile.isel(lead_time=idx), ds_model["level"], label=f"Model: Lead {lead}h", linewidth=2)

    if ds_ref is not None and metric in ds_ref:
        ref_profile = ds_ref[metric].mean(dim="lead_time")
        ax.plot(ref_profile, ds_ref["level"], label="Reference Mean", linewidth=2, linestyle='--')

    ax.invert_yaxis()
    ax.set_yticks([1000, 850, 700, 500, 300, 100, 1])
    ax.set_ylim(bottom=1000)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.4)
    ax.set_title(f"{metric.replace('_', ' ').capitalize()}\nMean over lat/lon", fontsize=16)
    ax.set_xlabel(f"{metric} mean")
    ax.set_ylabel("Pressure Level (hPa)")
    ax.set_xscale("log")
    ax.legend()
    plt.tight_layout()
    plot_path = os.path.join(SUMMARY_PLOT_DIR, f"{metric}_vertical_leadtime_profile.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()

def process_all():
    summary_model, summary_ref = load_summary_outputs()
    full_model_data, full_ref_data = load_fullfields_outputs()

    for metric in METRICS:
        plot_vertical_leadtime_profile(summary_model, summary_ref, metric)

        for (lead_time, ds_model), (_, ds_ref) in zip(full_model_data, full_ref_data):
            plot_path_model = os.path.join(FULLFIELDS_PLOT_DIR, f"{metric}_lead{lead_time}_model.png")
            plot_path_ref = os.path.join(FULLFIELDS_PLOT_DIR, f"{metric}_lead{lead_time}_ref.png")
            plot_spatial_slice(ds_model, metric, lead_time, level=500, title_prefix="Model", plot_path=plot_path_model)
            plot_spatial_slice(ds_ref, metric, lead_time, level=500, title_prefix="Reference", plot_path=plot_path_ref)

if __name__ == "__main__":
    process_all()

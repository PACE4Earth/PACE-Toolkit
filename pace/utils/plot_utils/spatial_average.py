import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr


def plot_spatial_averages(
    var_store: dict,
    coords: dict,
    save_dir: str = ".",
    dataset_name: str = "model",
):
    """
    Plot spatial averages (mean over idx and level) for all variables
    with dims ['idx', 'level', 'lat', 'lon'].

    Parameters
    ----------
    var_store : dict
        Dictionary {var_name: xr.DataArray} with dims ['idx','level','lat','lon'].
    coords : dict
        Dictionary with arrays for ['lat','lon','level','base_time','lead_time'].
    save_dir : str, default="."
        Directory to save plots.
    dataset_name : str, default="model"
        Used in filenames and figure titles.
    """

    out_dir = os.path.join(save_dir, "spatial_averages")
    os.makedirs(out_dir, exist_ok=True)

    lats = coords["lat"]
    lons = coords["lon"]
    Lon, Lat = np.meshgrid(lons, lats)

    for variable, da in var_store.items():
        # Skip variables not matching required dims
        if not all(dim in da.dims for dim in ["idx", "level", "lat", "lon"]):
            print(f"Skipping '{variable}' (dims: {list(da.dims)})")
            continue

        # Mean over idx and level → dims (lat, lon)
        da_mean = da.mean(dim=["idx", "level"], skipna=True)
        field = da_mean.values

        fig, ax = plt.subplots(figsize=(12, 6))
        pcm = ax.pcolormesh(
            Lon,
            Lat,
            field,
            cmap="viridis",
            norm=colors.LogNorm(vmin=1e-3, vmax=np.nanmax(field)),
        )
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label(variable.replace("_", " ").capitalize(), fontsize=14)

        ax.set_title(
            f"{dataset_name.capitalize()}: {variable.replace('_', ' ').capitalize()} "
            f"(Mean over time + levels)",
            fontsize=16,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        def add_degree_symbols(ticks):
            return [f"{tick:.1f}°" for tick in ticks]

        ax.set_xticklabels(add_degree_symbols(ax.get_xticks()))
        ax.set_yticklabels(add_degree_symbols(ax.get_yticks()))

        plt.tight_layout()
        filename = f"{dataset_name}_{variable}.png"
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved: {path}")

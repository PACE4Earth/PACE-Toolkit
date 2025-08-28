import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr


def plot_spatial_slice(
    var_store: dict,
    coords: dict,
    variable: str,
    level: float = 850,
    samples: int = 1,
    save_dir: str = ".",
    dataset_name: str = "model",
):
    """
    Plot spatial slices for the first `samples` from var_store for a given variable and pressure level.

    Parameters
    ----------
    var_store : dict
        Dictionary {var_name: xr.DataArray} with dims ['idx', 'level', 'lat', 'lon']
        plus coordinates ['base_time', 'lead_time'].
    coords : dict
        Dictionary with arrays for ['lat', 'lon', 'level', 'base_time', 'lead_time'].
    variable : str
        Variable name to plot.
    level : float, default=850
        Pressure level (hPa) to slice. Nearest level will be used if not exact.
    samples : int, default=1
        Number of samples (by `idx`) to plot.
    save_dir : str, default="."
        Directory to save plots.
    dataset_name : str, default="model"
        Used in filenames and figure titles.
    """

    if variable not in var_store:
        print(f"Variable '{variable}' not found in dataset '{dataset_name}'. Skipping.")
        return

    da = var_store[variable]

    if "level" not in da.dims:
        print(f"Variable '{variable}' does not have a 'level' dimension. Skipping.")
        return

    # Select nearest pressure level
    da_level = da.sel(level=level, method="nearest")
    selected_level = float(da_level.level.values)

    # Coordinates for plotting
    lats = coords["lat"]
    lons = coords["lon"]
    Lon, Lat = np.meshgrid(lons, lats)

    out_dir = os.path.join(save_dir, "spatial_slices")
    os.makedirs(out_dir, exist_ok=True)

    # Restrict number of samples
    if "idx" not in da.dims:
        print(f"Variable '{variable}' does not have an 'idx' dimension. Skipping.")
        return
    idxs = da["idx"].values[:samples]

    for i in idxs:
        slice_data = da_level.sel(idx=i).values  # shape (lat, lon)
        base_time = coords["base_time"][int(i)] if "base_time" in coords else None
        lead_time = coords["lead_time"][int(i)] if "lead_time" in coords else None

        fig, ax = plt.subplots(figsize=(12, 6))
        pcm = ax.pcolormesh(
            Lon,
            Lat,
            slice_data,
            cmap="viridis",
            norm=colors.LogNorm(vmin=1e-3, vmax=np.nanmax(slice_data)),
        )
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label(variable.replace("_", " ").capitalize(), fontsize=14)

        # Format title
        if base_time is not None:
            base_time_str = np.datetime_as_string(base_time, unit="m")
        else:
            base_time_str = "N/A"

        if lead_time is not None:
            lead_hours =  lead_time.astype('timedelta64[h]').astype(int)
        else:
            lead_hours = -1

        ax.set_title(
            f"{dataset_name.capitalize()}: {variable.replace('_', ' ').capitalize()} | "
            f"Base: {base_time_str} | Level: {int(selected_level)} hPa | Lead: {lead_hours}h",
            fontsize=16,
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        def add_degree_symbols(ticks):
            return [f"{tick:.1f}°" for tick in ticks]

        ax.set_xticklabels(add_degree_symbols(ax.get_xticks()))
        ax.set_yticklabels(add_degree_symbols(ax.get_yticks()))

        plt.tight_layout()
        filename = (
            f"{dataset_name}_{variable}_level{int(selected_level)}_"
            f"lead{lead_hours}h.png"
        )
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        print(f"Saved: {path}")

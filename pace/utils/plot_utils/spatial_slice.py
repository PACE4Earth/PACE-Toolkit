import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

def plot_spatial_slice(
    ds_store,
    coords,
    variable,
    level,
    samples=1,
    geopotential=None,
    geopotential_level=None,
    save_dir=".",
    dataset_name="model"
):
    """
    Plot spatial slices for first `samples` from ds_store for given variable and pressure level.
    Optionally overlays geopotential contours.

    Parameters:
    - ds_store: dict with keys like (var_name, base_time, lead_time) -> np.ndarray
    - coords: dict with 'lat', 'lon', 'pressure_levels' arrays
    - variable: str, variable name to plot
    - level: int or float, pressure level (hPa) to slice at
    - samples: int, number of samples to plot
    - geopotential: dict like ds_store for geopotential (optional)
    - geopotential_level: pressure level for geopotential contours (optional)
    - save_dir: path to save plots
    - dataset_name: str, used in filenames
    """

    lats = coords['lat']
    lons = coords['lon']
    levels = coords['pressure_levels']

    level_idx = np.abs(levels - level).argmin()
    selected_level = levels[level_idx]

    Lon, Lat = np.meshgrid(lons, lats)

    filtered_items = [(k, v) for k, v in ds_store.items() if k[0] == variable]
    filtered_items.sort(key=lambda x: (x[0][1], x[0][2]))

    n_samples = min(samples, len(filtered_items))

    out_dir = os.path.join(save_dir, "spatial_slices")
    os.makedirs(out_dir, exist_ok=True)

    for i in range(n_samples):
        (var_name, base_time, lead_time), data = filtered_items[i]

        if data.ndim == 4:
            slice_data = data[0, level_idx, :, :]
        elif data.ndim == 3:
            slice_data = data[level_idx, :, :]
        else:
            raise ValueError(f"Unexpected data ndim={data.ndim} for variable {var_name}")

        fig, ax = plt.subplots(figsize=(12, 6))
        pcm = ax.pcolormesh(
            Lon, Lat, slice_data, cmap='viridis',
            norm=colors.LogNorm(vmin=1e-3, vmax=np.max(slice_data))
        )
        cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
        cbar.set_label(f"{variable.replace('_', ' ').capitalize()}", fontsize=14)

        if geopotential and geopotential_level is not None:
            geo_key = ('geopotential', base_time, lead_time)
            if geo_key in geopotential:
                geo_data = geopotential[geo_key]
                if geo_data.ndim == 4:
                    geo_slice = geo_data[0, np.abs(levels - geopotential_level).argmin(), :, :]
                elif geo_data.ndim == 3:
                    geo_slice = geo_data[np.abs(levels - geopotential_level).argmin(), :, :]
                else:
                    raise ValueError("Unexpected geopotential data ndim")

                contour_levels = np.linspace(np.min(geo_slice), np.max(geo_slice), 10)
                cs = ax.contour(Lon, Lat, geo_slice, levels=contour_levels, colors='white', linewidths=1)
                ax.clabel(cs, inline=1, fontsize=10, fmt="%.0f")

        if hasattr(base_time, 'astype'):
            base_time_str = np.datetime_as_string(base_time, unit='m')
        else:
            base_time_str = str(base_time)

        ax.set_title(
            f"{dataset_name.capitalize()}: {variable.replace('_', ' ').capitalize()} | Base: {base_time_str} | Level: {int(selected_level)} hPa | Lead: {lead_time}h",
            fontsize=16
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        def add_degree_symbols(ticks):
            return [f"{tick:.1f}°" for tick in ticks]

        ax.set_xticklabels(add_degree_symbols(ax.get_xticks()))
        ax.set_yticklabels(add_degree_symbols(ax.get_yticks()))

        plt.tight_layout()
        filename = f"{dataset_name}_{variable}_level{int(selected_level)}_lead{lead_time}h.png"
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()

        print(f"Saved: {path}")

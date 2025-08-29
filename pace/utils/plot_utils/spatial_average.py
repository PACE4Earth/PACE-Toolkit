import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import gc


def plot_spatial_averages(var_store: dict, coords: dict, save_dir: str = ".", dataset_name: str = "model"):
    """
    Memory-efficient plotting of spatial averages (mean over idx and level) for all variables
    with dims ['idx','level','lat','lon'] by computing mean incrementally.
    """

    out_dir = os.path.join(save_dir, "spatial_averages")
    os.makedirs(out_dir, exist_ok=True)

    lats = coords["lat"]
    lons = coords["lon"]

    for variable, da in var_store.items():
        if not all(dim in da.dims for dim in ["idx", "level", "lat", "lon"]):
            print(f"Skipping '{variable}' (dims: {list(da.dims)})")
            continue

        # Compute mean over idx incrementally to save memory
        n_idx = da.sizes["idx"]
        mean_accum = np.zeros((da.sizes["level"], da.sizes["lat"], da.sizes["lon"]), dtype=np.float64)
        count_accum = np.zeros_like(mean_accum)

        for i in range(n_idx):
            slice_i = da.isel(idx=i)
            valid_mask = ~np.isnan(slice_i.values)
            mean_accum[valid_mask] += slice_i.values[valid_mask]
            count_accum[valid_mask] += 1

        # Avoid division by zero
        count_accum[count_accum == 0] = np.nan
        mean_over_idx = mean_accum / count_accum

        # Mean over level
        mean_field = np.nanmean(mean_over_idx, axis=0)  # dims (lat, lon)

        # Downcast to float32 to save memory
        mean_field = mean_field.astype(np.float32)

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        im = ax.imshow(
            mean_field,
            origin="lower",
            cmap="viridis",
            interpolation="none",
            extent=[np.nanmin(lons), np.nanmax(lons), lats.min(), lats.max()],
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.5)
        cbar.set_label(variable)

        ax.set_title(
            f"{dataset_name.capitalize()}: {variable.replace('_',' ').capitalize()} (Mean over time + levels)",
            fontsize=16,
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}°"))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:.1f}°"))

        plt.tight_layout()
        filename = f"{dataset_name}_{variable}.png"
        path = os.path.join(out_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close(fig)

        # Free memory explicitly
        del mean_accum, count_accum, mean_over_idx, mean_field
        gc.collect()

        print(f"Saved: {path}")

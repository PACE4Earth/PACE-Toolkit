import os

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Dict, Tuple, Optional
import seaborn as sns
import xarray as xr

def plot_map(
        model_ds=None,
        ref_ds=None,
        model_name=None,
        ref_name=None,
        plots_dir=None,
        coords=None,
    ):

    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(9,9), sharex=True, sharey=True)
    
    lats = coords["lat"]
    lons = coords["lon"]
    lats = np.nan_to_num(lats, nan=0.0)
    lons = np.nan_to_num(lons, nan=0.0)
    Lon, Lat = np.meshgrid(lons, lats)
    
    for ax in axs.flatten():
        # ax.set_xticks(model_ds['lon'].values)
        # ax.set_yticks(model_ds['lat'].values)
        ax.set_aspect(1)
    
        
    model_data = model_ds['correlation_map'].values
    ref_data = ref_ds['correlation_map'].values 
    
    _, c, _, _ = model_data.shape
    rows, cols = np.triu_indices(c, k=1)
    model_tri_data = model_data[rows, cols, :, :]
    ref_tri_data = ref_data[rows, cols, :, :]
        
    v, h, w = model_tri_data.shape

    axs[0, 0].set_title(model_name)
    axs[0, 1].set_title(ref_name)
    axs[0, 2].set_title(f"{model_name}-{ref_name}")

    # Now, loop through all rows to set the variable titles and plot the data
    for i in range(v):
        new_title_row = f'{model_ds.var_1.isel(var_1=rows[i]).values}\n{model_ds.var_2.isel(var_2=cols[i]).values}'

        # For the first row, append the variable info to the existing titles
        if i == 0:
            axs[i, 0].set_title(f'{axs[i, 0].get_title()}\n{new_title_row}')
            axs[i, 1].set_title(f'{axs[i, 1].get_title()}\n{new_title_row}')
            axs[i, 2].set_title(f'{axs[i, 2].get_title()}\n{new_title_row}')
        
        # For all other rows, just set the variable title (no model/ref name)
        else:
            axs[i, 0].set_title(new_title_row)
            axs[i, 1].set_title(new_title_row)
            axs[i, 2].set_title(new_title_row)

        # Plotting code remains the same
        im = axs[i, 0].pcolormesh(Lon, Lat, model_tri_data[i], cmap='coolwarm', vmin=-1, vmax=1)
        im = axs[i, 1].pcolormesh(Lon, Lat, ref_tri_data[i], cmap='coolwarm', vmin=-1, vmax=1)
        im = axs[i, 2].pcolormesh(Lon, Lat, (model_tri_data[i] - ref_tri_data[i]), cmap='coolwarm', vmin=-1, vmax=1)
  
        divider = make_axes_locatable(axs[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.1)

        # Add the colorbar to the new axes
        fig.colorbar(im, cax=cax)
  
    # print(plots_dir)
  
    plt.tight_layout()
    # plt.savefig(os.path.join(plots_dir, f'correlation_map_{model_name}.png'))
    # plt.savefig(os.path.join(plots_dir, f'correlation_map_rea2.png'))
    plt.savefig(os.path.join(plots_dir, f'correlation_map_{model_name}.png'))
    
    print()
    
    return
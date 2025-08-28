import os

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import seaborn as sns
import xarray as xr

def plot_corr_time_series(
        model_ds=None,
        ref_ds=None,
        model_name=None,
        ref_name=None,
        plots_dir=None,
        coords=None,
    ):

    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(6,9), sharex=True, sharey=True)
    
    fig.suptitle(f'corr({model_name}) - corr({ref_name})')
    
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
    
    # print(model_tri_data.shape)
    
    v, h, w = model_tri_data.shape
    
    for i in range(v):
        axs[i].set_title(f'{model_ds.var_1.isel(var_1=rows[i]).values}\n{model_ds.var_2.isel(var_2=cols[i]).values}')
        im = axs[i].pcolormesh(
            Lon,
            Lat,
            model_tri_data[i]-ref_tri_data[i],
        )
        plt.colorbar(im, ax=axs[i])
  
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'correlation_map.png'))
    
    return
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
    ):
    
    dss = [ds for ds in [model_ds, ref_ds] if ds is not None]
    names = [name for name in [model_name, ref_name] if name is not None]
    
    fig, axs = plt.subplots(ncols=1, nrows=3, figsize=(9,9), sharex=True)
    
    for ax in axs.flatten():
        ax.set_ylim(-1, 1)
        ax.grid(which='both', linestyle='-.')
    
    for it, (name, ds) in enumerate(zip(names, dss)):
                           
        # ds = ds.where(ds.lead_time==np.timedelta64(24, 'h'))

        valid_times = ds['base_time'].values + ds['lead_time'].values
        
        # print('DEBUG___________________________________________')
        # for bt, lt in zip(ds['base_time'].values, ds['lead_time'].values):
        #     print('valid_time <- base_time + lead_time', bt+lt, bt, lt)
        
        sample = ds['corr_column']
        data = sample.values
        data = sample.values
        _, c, _ = data.shape
        
        rows, cols = np.triu_indices(c, k=1)
        time_series_data = data[:, rows, cols]
        
        n, e = time_series_data.shape
        
        for i in range(e):
            axs[i].scatter(
                valid_times, 
                time_series_data[:, i], 
                label=f'{name}: ({ds.var_1.isel(var_1=rows[i]).values}, {ds.var_2.isel(var_2=cols[i]).values})', 
                alpha=0.4
            )
        
    plt.xlabel('valid time')
    
    for ax in axs.flatten():
        ax.legend(title='Matrix Indices (i, j)', loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'correlation.png'))
    
    return
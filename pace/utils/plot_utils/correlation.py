from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

import numpy as np
import torch
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns


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
    plt.savefig(os.path.join(plots_dir, f'correlation_{model_name}.png'))
    
    return

def visualize(model_hist, ref_hist, key, ax):
    """Visualizes the histogram for a given key using its specific range."""

    fig = ax[0].get_figure()

    model_data = model_hist['tensor']
    model_data = (model_data / model_data.sum()).clamp(min=1e-6).cpu().numpy()
    # model_data = model_data.clamp(min=1e-8).cpu().numpy()
    
    ref_data = ref_hist['tensor']
    ref_data = (ref_data / ref_data.sum()).clamp(min=1e-6).cpu().numpy()
    # ref_data = ref_data.clamp(min=1e-8).cpu().numpy()
    
    
    range_x, range_y = model_hist['range']
    extent = [range_x[0], range_x[1], range_y[0], range_y[1]]

    x = np.linspace(range_x[0], range_x[1], model_data.shape[1])
    y = np.linspace(range_y[0], range_y[1], model_data.shape[0])

    # Define the contour levels
    # Using LogNorm() suggests that the data spans several orders of magnitude, so we create levels on a logarithmic scale.
    # levels = np.logspace(np.log10(model_data.min()), np.log10(model_data.max()), 8)
    # levels = np.logspace(-6.01, -2.5, 8)
    levels = np.array([10**(-6.01), 10**(-6), 10**(-5.), 10**(-4.5), 1e-4, 10**(-3.5), 1e-3, 10**(-2.5), 1e-2])

    # Create the contourf plot
    im = ax[0].contourf(
        x, y, model_data,
        levels=levels,
        cmap='magma',
        # colors='white',
        norm=colors.LogNorm(),
    )
    
    ax[0].contour(
        x, 
        y,
        ref_data,
        levels=levels,
        # colors='w', # or use cmap='jet' and pass norm again
        cmap='binary',
        linewidths=1.,
        alpha=0.5,
        norm=colors.LogNorm()
    )
    
    diff = model_data-ref_data
    
    max_diff = max(diff.max(), np.abs(diff).max())
    
    log_norm = colors.LogNorm(vmin=1e-5, vmax=1e-2)
    
    im2 = ax[1].pcolormesh(
        x, 
        y,
        (diff>0).astype(int)*(diff+1e-5),
        cmap='Reds',
        # vmin=-np.log10(max_diff),
        # vmin=1e-8,
        # vmax=max_diff,
        norm=log_norm,
    )
    
    im3 = ax[1].pcolormesh(
        x, 
        y,
        (diff<0).astype(int)*(-diff+1e-5),
        cmap='Blues',
        # vmin=-np.log10(max_diff),
        # vmin=1e-8,
        # vmax=max_diff,
        norm=log_norm,
    )
    
    var_x, var_y = tuple(key.split('.'))
    for a in ax.flatten():
        a.set_xlabel(var_x)
        a.set_ylabel(var_y)
        # a.set_title(f"Histogram for {key}")
    fig.colorbar(im, ax=ax[0], label='Prob. density') 
    fig.colorbar(im2, ax=ax[1], label='Diff. prob. density') 
    fig.colorbar(im3, ax=ax[1]) 
    
    return fig, ax

def plot_bivar_hist(model_name, ref_name, outputs_dir, plots_dir, valid_times):
    
    for f in os.listdir(outputs_dir):
        if f == model_name:
            model_dir = os.path.join(outputs_dir, f)
        elif f == ref_name:
            ref_dir = os.path.join(outputs_dir, f)
            
    fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(9, 9))
    if len(valid_times)==1:
        fig.suptitle(valid_times[0].astype('datetime64[h]'))   
            
    for i, (m, r) in enumerate(zip(os.listdir(model_dir), os.listdir(ref_dir))):
        
        key = m[ m.find('.') + 1 : m.rfind('.') ]
        
        m_h = torch.load(os.path.join(model_dir, m))
        r_h = torch.load(os.path.join(ref_dir, r))
        
        visualize(m_h, r_h, key, axs[i, :])
        
        print(i)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'bivar_hist_{model_name}.png'))
    
    return
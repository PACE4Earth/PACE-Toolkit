import os

import numpy as np
import zarr
import xarray as xr

import pandas as pd # Needed for timedelta conversion
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

import datetime

def main():
    
    plot_dir = './outputs'
    
    zarr_paths = [
        # '/p/scratch/hclimrep/vozar2/outputs_cluster/corrdiff_pred.zarr',
        # '/p/scratch/hclimrep/vozar2/outputs_cluster/corrdiff_reference.zarr',
        # '/p/scratch/hclimrep/vozar2/outputs_cluster/graphcast.zarr',
        '/p/scratch/hclimrep/vozar2/outputs_cluster/era5.zarr'
    ]
    
    names = [
        # 'CorrDiff',
        # 'REA2',
        # 'GraphCast',
        'ERA5',
    ]

    # with xr.open_zarr(zarr_path, consolidated=False) as ds:
    #     print(ds, '\n__________\n')
        
    #     # Using .values is a good practice to avoid potential coordinate conflicts
    #     sorting_indices = ds.base_time.argsort()
    #     ds = ds.isel(idx=sorting_indices.values)
        
    #     sample = ds['corr_column']
    #     print(sample, '\n__________\n')
        
    #     data = sample.values
    #     print(type(data), type(data[0]), data.shape)
            
    #     _, c, _ = data.shape
        
    #     rows, cols = np.triu_indices(c, k=1)
    #     time_series_data = data[:, rows, cols]
        
    #     n, e = time_series_data.shape
    #     print(n, e)

    #     # --- NEW SETUP CODE STARTS HERE ---
        
    #     # 1. Get lead_time values and convert to numeric hours
    #     lead_time_values = ds['lead_time'].values
    #     lead_time_hours = lead_time_values / np.timedelta64(1, 'h')
        
        
        
    #     print(lead_time_hours)

    #     # 2. Normalize lead_time to a visible alpha range [0.2, 1.0]
    #     min_lt, max_lt = np.nanmin(lead_time_hours), np.nanmax(lead_time_hours)
    #     norm = mcolors.Normalize(vmin=min_lt, vmax=max_lt)
    #     alphas = 0.1 + 0.9 * norm(lead_time_hours) # This is our array of alpha values, shape (n,)
        
    #     # 3. Get the default color cycle from matplotlib
    #     prop_cycle = plt.rcParams['axes.prop_cycle']
    #     color_cycle = prop_cycle.by_key()['color']
        
    #     # --- NEW SETUP CODE ENDS HERE ---

    #     fig = plt.figure(figsize=(9,4))
        
    #     # This is your original plotting loop, with one line inside changed
    #     for i in range(e):
    #         # --- MODIFICATIONS INSIDE THE LOOP ---
            
    #         # A. Get the base color for this series from the color cycle
    #         base_color = color_cycle[i % len(color_cycle)]
            
    #         # B. Create an RGBA color array for this specific series
    #         # Start with the base RGB color and repeat it 'n' times
    #         series_colors = np.array([mcolors.to_rgb(base_color)] * n)
            
    #         # C. Add the alpha channel as the 4th column
    #         # The result is an (n, 4) array for this series' points
    #         series_colors_with_alpha = np.insert(series_colors, 3, alphas, axis=1)

    #         # D. Use the new color array in your scatter plot call
    #         plt.scatter(
    #             time_series_data[:, i], 
    #             range(n), 
    #             label=f'({rows[i]}, {cols[i]})', 
    #             color=series_colors_with_alpha
    #         ) # MODIFIED: Use 'color' instead of default
            
    #         # plt.hist(time_series_data[:, i],
    #         #     bins=100,  # You can specify the number of bins
    #         #     label=f'({rows[i]}, {cols[i]})',
    #         #     color=series_colors_with_alpha[i],
    #         #     width=0.01,
    #         #     density=True
    #         # )
            
    #         # --- END OF MODIFICATIONS ---

    #     plt.xlabel("Value")
    #     plt.ylabel("Frequency")
    #     plt.title("Histogram of Time Series Data")
    #     plt.legend()
    #     plt.grid(True, linestyle='--', alpha=0.5)
    #     plt.savefig(os.path.join(plot_dir, 'sample_edited.png'))
    #     plt.close()

    # print("Edited plot has been saved.")
    
    if 'ERA5' in names: nrows = 6
    elif 'REA2' in names: nrows = 3
    
    fig, axs = plt.subplots(ncols=1, nrows=nrows, figsize=(9,9), sharex=True)
    
    for ax in axs.flatten():
        ax.set_ylim(-1, 1)
        ax.grid(which='both', linestyle='-.')
    
    for it, (name, zarr_path) in enumerate(zip(names, zarr_paths)):
        
        with xr.open_zarr(zarr_path) as ds:
            
            print(ds)
            
            # ds = ds.where(ds.lead_time==np.timedelta64(24, 'h'))
            
            # Using .values is a good practice to avoid potential coordinate conflicts
            # sorting_indices = ds.base_time.argsort()
            # ds = ds.isel(idx=sorting_indices.values)
            
            valid_times = ds['base_time'].values+ds['lead_time'].values
            print(valid_times)
            
            sample = ds['corr_column']
                
            print(sample, '\n__________\n')
            
            data = sample.values
            print(type(data), type(data[0]), data.shape)
            
        data = sample.values
        print(type(data), type(data[0]), data.shape)
            
        _, c, _ = data.shape
        
        rows, cols = np.triu_indices(c, k=1)
        print(rows, cols)
        time_series_data = data[:, rows, cols]
        
        n, e = time_series_data.shape
        print(n, e)
        
        
        for i in range(e):
            axs[i].scatter(
                valid_times, 
                time_series_data[:, i], 
                label=f'{name}: ({ds.var_1.isel(var_1=rows[i]).values}, {ds.var_2.isel(var_2=cols[i]).values})', 
                alpha=0.4
            )
        
    plt.xlabel('valid time')
    # plt.ylabel('correlation coeff')
    
    for ax in axs.flatten():
        ax.legend(title='Matrix Indices (i, j)', loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'sample_{names[-1]}.png'))
    
    return 0
    
if __name__=="__main__":
    
    main()
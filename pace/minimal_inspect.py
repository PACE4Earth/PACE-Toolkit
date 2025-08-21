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
    
    base_path = '/p/scratch/hclimrep/vozar2/outputs_cluster/graphcast.zarr'
    plot_dir = './outputs'

    with xr.open_zarr(base_path, consolidated=False) as ds:
        print(ds, '\n__________\n')
        
        # Using .values is a good practice to avoid potential coordinate conflicts
        sorting_indices = ds.base_time.argsort()
        ds = ds.isel(idx=sorting_indices.values)
        
        sample = ds['correlation']
        print(sample, '\n__________\n')
        
        data = sample.values
        print(type(data), type(data[0]), data.shape)
            
        _, c, _ = data.shape
        
        rows, cols = np.triu_indices(c, k=1)
        time_series_data = data[:, rows, cols]
        
        n, e = time_series_data.shape
        print(n, e)

        # --- NEW SETUP CODE STARTS HERE ---
        
        # 1. Get lead_time values and convert to numeric hours
        lead_time_values = ds['lead_time'].values
        lead_time_hours = lead_time_values / np.timedelta64(1, 'h')

        # 2. Normalize lead_time to a visible alpha range [0.2, 1.0]
        min_lt, max_lt = np.nanmin(lead_time_hours), np.nanmax(lead_time_hours)
        norm = mcolors.Normalize(vmin=min_lt, vmax=max_lt)
        alphas = 0.1 + 0.9 * norm(lead_time_hours) # This is our array of alpha values, shape (n,)
        
        # 3. Get the default color cycle from matplotlib
        prop_cycle = plt.rcParams['axes.prop_cycle']
        color_cycle = prop_cycle.by_key()['color']
        
        # --- NEW SETUP CODE ENDS HERE ---

        fig = plt.figure(figsize=(9,4))
        
        # This is your original plotting loop, with one line inside changed
        for i in range(e):
            # --- MODIFICATIONS INSIDE THE LOOP ---
            
            # A. Get the base color for this series from the color cycle
            base_color = color_cycle[i % len(color_cycle)]
            
            # B. Create an RGBA color array for this specific series
            # Start with the base RGB color and repeat it 'n' times
            series_colors = np.array([mcolors.to_rgb(base_color)] * n)
            
            # C. Add the alpha channel as the 4th column
            # The result is an (n, 4) array for this series' points
            series_colors_with_alpha = np.insert(series_colors, 3, alphas, axis=1)

            # D. Use the new color array in your scatter plot call
            # plt.scatter(range(n), 
            #             time_series_data[:, i], 
            #             label=f'({rows[i]}, {cols[i]})', 
            #             color=series_colors_with_alpha
            # ) # MODIFIED: Use 'color' instead of default
            
            plt.hist(time_series_data[:, i],
                bins=20,  # You can specify the number of bins
                label=f'({rows[i]}, {cols[i]})',
                color=series_colors_with_alpha[i],
                width=0.005
            )
            
            # --- END OF MODIFICATIONS ---

        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.title("Histogram of Time Series Data")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(plot_dir, 'sample_edited.png'))
        plt.close()

    print("Edited plot has been saved.")
    
    # with xr.open_zarr(base_path, consolidated=False) as ds:
    #     print(ds, '\n__________\n')
        
    #     sorting_indices = ds.base_time.argsort()
    #     print("\nSorting Indices:", sorting_indices.values)
    #     ds = ds.isel(idx=sorting_indices)
        
    #     sample = ds['correlation_corrdiff']
    #     print(sample, '\n__________\n')
        
    # data = sample.values
    # print(type(data), type(data[0]), data.shape)
        
    # _, c, _ = data.shape
    
    # rows, cols = np.triu_indices(c, k=1)
    # time_series_data = data[:, rows, cols]
    
    # n, e = time_series_data.shape
    # print(n, e)
    
    # fig = plt.figure(figsize=(9,4))
    
    # for i in range(e):
    #     plt.scatter(range(n), time_series_data[:, i], label=f'({rows[i]}, {cols[i]})')
    
    # plt.legend(title='Matrix Indices (i, j)', loc='upper left')
    # plt.savefig(os.path.join(plot_dir, 'sample.png'))
    
    return 0
    
if __name__=="__main__":
    
    main()
import os

import torch
from torch import nn
from torch.nn import functional as F
        
import matplotlib.pyplot as plt
import matplotlib.colors as colors

try:
    from .operators import standardize
except Exception as e:
    from operators import standardize

WIND_MIN = -30      # m/s
WIND_MAX = 30       # m/s
T_MIN = 220         # K
T_MAX = 325         # K
MSLP_MIN = 95000    # Pa
MSLP_MAX = 107000   # Pa
VMAX_MIN = 0
VMAX_MAX = 40
TP_MIN = 0
TP_MAX = 200
    
# range=(
#     (-1.0, 1.0), # t2m, u10m 
#     (-1.0, 1.0), # t2m, v10m 
#     (-1.0, 1.0), # t2m, mslp 
#     (-1.0, 1.0), # u10m, v10m 
#     (-1.0, 1.0), # u10m, mslp 
#     (-1.0, 1.0), # v10m, mslp
# )
# HISTOGRAM_CONFIG below is the generic config associated with the variable ranges and bins defined at the header.
HISTOGRAM_CONFIG = {
    '2m_temperature.10m_u_component_of_wind': {
        'bins': 64,
        'range': ((T_MIN, T_MAX), (WIND_MIN, WIND_MAX))
    },
    '2m_temperature.10m_v_component_of_wind': {
        'bins': 64,
        'range': ((T_MIN, T_MAX), (WIND_MIN, WIND_MAX))
    },
    '2m_temperature.mean_sea_level_pressure': {
        'bins': 64,
        'range': ((T_MIN, T_MAX), (MSLP_MIN, MSLP_MAX))
    },
    '10m_u_component_of_wind.10m_v_component_of_wind': {
        'bins': 64,
        'range': ((WIND_MIN, WIND_MAX), (WIND_MIN, WIND_MAX))
    },
    '10m_u_component_of_wind.mean_sea_level_pressure': {
        'bins': 64,
        'range': ((WIND_MIN, WIND_MAX), (MSLP_MIN, MSLP_MAX))
    },
    '10m_v_component_of_wind.mean_sea_level_pressure': {
        'bins': 64,
        'range': ((WIND_MIN, WIND_MAX), (MSLP_MIN, MSLP_MAX))
    },
    '2m_temperature.total_precipitation': {
        'bins': 64,
        'range': ((T_MIN, T_MAX), (TP_MIN, TP_MAX))
    }, 
    '2m_temperature.vmax_10m': {
        'bins': 64,
        'range': ((T_MIN, T_MAX), (VMAX_MIN, VMAX_MAX))
    }, 
    'total_precipitation.vmax_10m': {
        'bins': 64,
        'range': ((TP_MIN, TP_MAX), (VMAX_MIN, VMAX_MAX))
    }, 
}

class GenericHistogram(nn.Module):
    def __init__(
        self, 
        grid,  
        pairs=None,
        histogram_config=None,
        ):
        super().__init__()
        self.histograms = {}
        self.device = os.getenv('DEVICE')
        self.var_ranges = {
            '2m_temperature': (T_MIN, T_MAX),
            '10m_u_component_of_wind': (WIND_MIN, WIND_MAX),
            '10m_v_component_of_wind': (WIND_MIN, WIND_MAX),
            'mean_sea_level_pressure': (MSLP_MIN, MSLP_MAX),
            'vmax_10m': (VMAX_MIN, VMAX_MAX),
            'total_precipitation': (TP_MIN, TP_MAX)
        }
        self.default_bins = 64
        self.corr = None
        self.pairs = pairs
        
        # Use provided pairs or fallback to config-derived pairs
        # if pairs is not None:
        #     self.pairs = pairs
        # else:
        #     config = histogram_config if histogram_config is not None else HISTOGRAM_CONFIG
        #     self.pairs = []
        #     for key in config.keys():
        #         var_x, var_y = key.split('.', 1)
        #         self.pairs.append((var_x, var_y))

    def add_histogram(self, var_x, var_y, bins=None, range_x=None, range_y=None, key=None):
        
        if key is None:
            key = f"{var_x}.{var_y}"
        
        bins = bins or self.default_bins
        bins_x, bins_y = (bins, bins) if isinstance(bins, int) else (bins[1], bins[0])
        range_x = range_x or self.var_ranges.get(var_x)
        range_y = range_y or self.var_ranges.get(var_y)
        if range_x is None or range_y is None:
            raise ValueError(f"Ranges for {var_x} or {var_y} not specified.")
        self.histograms[key] = {
            'tensor': torch.zeros((bins_y, bins_x), dtype=torch.long),
            'bins': (bins_x, bins_y),
            'range': (range_x, range_y),
            'vars': (var_x, var_y),
        }

    def evaluate_corr(self):
        
        mean_corr = self.corr[1:].mean(dim=0)
        std_corr = self.corr[1:].std(dim=0)
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)
        
        for ax in axs.flatten():
            ax.set_xticks([0.5, 1.5, 2.5, 3.5])
            ax.set_xticklabels(['t2m', 'u10m', 'v10', 'mslp'])
            ax.set_yticks([0.5, 1.5, 2.5, 3.5])
            ax.set_yticklabels(['t2m', 'u10m', 'v10', 'mslp'])
        
        axs[0].set_title('Corr. mean')
        axs[1].set_title('Corr. std')
        
        im1 = axs[0].pcolormesh(mean_corr, vmin=-1, vmax=1, cmap='seismic')
        # Add a colorbar to the first subplot
        fig.colorbar(im1, ax=axs[0])

        # Do the same for the second subplot
        im2 = axs[1].pcolormesh(std_corr, vmin=0, vmax=0.5, cmap='hot')
        fig.colorbar(im2, ax=axs[1])
        
        fig.tight_layout()
        
        return fig, axs

    def to(self, device):
        """Moves all managed histograms to the specified device."""
        self.device = device
        for key in self.histograms:
            self.histograms[key]['tensor'] = self.histograms[key]['tensor'].to(self.device)
        return self

    def update_histogram(self, key, x_coords, y_coords):
        """
        Updates the histogram for a predefined key.
        """
        if key not in self.histograms:
            raise KeyError(f"Key '{key}' not found. It was not defined in the initial config.")

        hist_info = self.histograms[key]
        bins_x, bins_y = hist_info['bins']
        range_x, range_y = hist_info['range']

        x_coords, y_coords = x_coords.to(self.device), y_coords.to(self.device)

        mask = (x_coords >= range_x[0]) & (x_coords < range_x[1]) & \
               (y_coords >= range_y[0]) & (y_coords < range_y[1])
        
        x, y = x_coords[mask], y_coords[mask]

        x_indices = ((x - range_x[0]) / (range_x[1] - range_x[0]) * bins_x).long()
        y_indices = ((y - range_y[0]) / (range_y[1] - range_y[0]) * bins_y).long()

        flat_indices = y_indices * bins_x + x_indices
        
        hist_increment = torch.bincount(
            flat_indices, 
            minlength=bins_y * bins_x
        ).view(bins_y, bins_x)
        
        hist_info['tensor'] += hist_increment

    def get_histogram(self, key):
        """Returns the histogram tensor for a specific key."""
        return self.histograms.get(key, {}).get('tensor')
    
    def evaluate(self, logger, comm):
        
        rank = comm.Get_rank()
        
        if not os.path.exists(f"{logger.path.split('.')[0]}"):
            os.mkdir(f"{logger.path.split('.')[0]}")
        
        for key, hist_info in self.histograms.items():
            
            this_path = os.path.join(f"{logger.path.split('.')[0]}", f"{rank}.{key}.pt")
            
            torch.save(hist_info, this_path)
        
        return ...

    def visualize(self, key, ax=None, cmap='hot'):
        """Visualizes the histogram for a given key using its specific range."""

        hist_info = self.histograms.get(key)
        if not hist_info:
            # This case is less likely now, but good practice to keep
            print(f"Warning: Histogram with key '{key}' not found.")
            return None, None

        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
        else:
            fig = ax.get_figure()

        data = hist_info['tensor']
        # norm
        data = (data / data.sum()).clamp(min=1e-8).cpu().numpy()
        
        range_x, range_y = hist_info['range']
        extent = [range_x[0], range_x[1], range_y[0], range_y[1]]

        im = ax.imshow(
            data, 
            cmap=cmap, 
            norm=colors.LogNorm(), 
            extent=extent, 
            origin='lower', 
            aspect='auto',
        )
        
        var_x, var_y = hist_info.get('vars', tuple(key.split('.', 1)))
        ax.set_xlabel(var_x)
        ax.set_ylabel(var_y)
        ax.set_title(f"Histogram for {key}")
        fig.colorbar(im, ax=ax, label='Prob. density') 
        
        return fig, ax

    def forward(self, sample):
        """
        Processes all tensors in the sample dictionary to compute their 
        covariance matrix.
        """

        processed_tensors = []
        variable_names = []
        for name, tensor in sample.items():
            if isinstance(tensor, torch.Tensor) and name in self.var_ranges.keys():
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim == 4:
                ############################################# this
                    tensor = tensor[0, [0]]
                
                tensor = standardize(tensor)
                    
                processed_tensors.append(tensor)
                variable_names.append(name)
         
        if self.pairs == None:
            self.pairs = [
                (variable_names[0], variable_names[1]),
                (variable_names[0], variable_names[2]),
                (variable_names[1], variable_names[2]),
            ]
                    
        data = torch.stack(processed_tensors, dim=0).contiguous()
        c, n, h, w = data.shape
        
        corr_column = []
        for jt in range(n):
            corr_item = torch.cov(data[:, jt, ...].reshape(c, -1))
            corr_column.append(corr_item.unsqueeze(0))
        
        corr_column = torch.cat(corr_column, dim=0)
        
        for var_x, var_y in self.pairs:
            
            try:
                key = f"{var_x}.{var_y}"
                
                if key not in self.histograms:
                    self.add_histogram(var_x, var_y)
                
                self.update_histogram(
                    key,
                    sample[var_x].flatten(),
                    sample[var_y].flatten(),
                )
            except Exception as e:
                ...
        
        return corr_column, variable_names
    
    def output_keys(self):
        return ['corr_column', 'var_names']

    # def forward(self, sample):
    #     z_temperature = standardize(sample['2m_temperature']).flatten()
    #     z_u10m = standardize(sample['10m_u_component_of_wind']).flatten()
    #     z_v10m = standardize(sample['10m_v_component_of_wind']).flatten()
    #     z_mslp = standardize(sample['mean_sea_level_pressure']).flatten()
        
    #     data = torch.stack([
    #         z_temperature,
    #         z_u10m,
    #         z_v10m,
    #         z_mslp,
    #     ])
        
    #     for var_x, var_y in self.pairs:
    #         key = f"{var_x}.{var_y}"
    #         if key not in self.histograms:
    #             self.add_histogram(var_x, var_y)
    #         self.update_histogram(
    #             key,
    #             sample[var_x].flatten(),
    #             sample[var_y].flatten(),
    #         )
    #     corr = torch.cov(data).unsqueeze(0)   
    #     self.corr = torch.cat([self.corr, corr], dim=0)     
    #     return corr

if __name__ == "__main__":
    import numpy as np
    os.environ['DEVICE'] = 'cpu'
    # Use scaled random normal samples for each variable
    sample = {
        '2m_temperature': torch.tensor(
            np.random.randn(128, 128) * ((T_MAX - T_MIN) / 6) + ((T_MAX + T_MIN) / 2),
            dtype=torch.float32
        ),
        '10m_u_component_of_wind': torch.tensor(
            np.random.randn(128, 128) * ((WIND_MAX - WIND_MIN) / 6) + ((WIND_MAX + WIND_MIN) / 2),
            dtype=torch.float32
        ),
        '10m_v_component_of_wind': torch.tensor(
            np.random.randn(128, 128) * ((WIND_MAX - WIND_MIN) / 6) + ((WIND_MAX + WIND_MIN) / 2),
            dtype=torch.float32
        ),
        'mean_sea_level_pressure': torch.tensor(
            np.random.randn(128, 128) * ((MSLP_MAX - MSLP_MIN) / 6) + ((MSLP_MAX + MSLP_MIN) / 2),
            dtype=torch.float32
        ),
    }
    # Instantiate GenericHistogram, optionally pass pairs or config
    metric = GenericHistogram(grid=None)
    corr = metric.forward(sample)
    
    key = '2m_temperature.10m_u_component_of_wind'
    
    fig, ax = metric.visualize(key)
    plt.savefig(f'corr_{key}.png')
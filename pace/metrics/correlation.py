import torch
from torch import nn
from torch.nn import functional as F
        
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from .operators import standardize

WIND_MIN = -30      # m/s
WIND_MAX = 30       # m/s
T_MIN = 220         # K
T_MAX = 325         # K
MSLP_MIN = 95000    # Pa
MSLP_MAX = 107000   # Pa

range=(
    (-1.0, 1.0), # t2m, u10m 
    (-1.0, 1.0), # t2m, v10m 
    (-1.0, 1.0), # t2m, mslp 
    (-1.0, 1.0), # u10m, v10m 
    (-1.0, 1.0), # u10m, mslp 
    (-1.0, 1.0), # v10m, mslp
)
HISTOGRAM_CONFIG = {
    't2m_u10m': {
        'bins': 64,
        'range': ((T_MIN, T_MAX), (WIND_MIN, WIND_MAX))
    },
    't2m_v10m': {
        'bins': 64,
        'range': ((T_MIN, T_MAX), (WIND_MIN, WIND_MAX))
    },
    't2m_mslp': {
        'bins': 64,
        'range': ((T_MIN, T_MAX), (MSLP_MIN, MSLP_MAX))
    },
    'u10m_v10m': {
        'bins': 64,
        'range': ((WIND_MIN, WIND_MAX), (WIND_MIN, WIND_MAX))
    },
    'u10m_mslp': {
        'bins': 64,
        'range': ((WIND_MIN, WIND_MAX), (MSLP_MIN, MSLP_MAX))
    },
    'v10m_mslp': {
        'bins': 64,
        'range': ((WIND_MIN, WIND_MAX), (MSLP_MIN, MSLP_MAX))
    },
}

class SampleWiseCorrelation(nn.Module):
    def __init__(
        self, 
        grid,  
        ):
        
        super().__init__()
        
        self.histograms = {}
        self.device = 'cpu'

        for key, settings in HISTOGRAM_CONFIG.items():
            bins = settings['bins']
            bins_y, bins_x = (bins, bins) if isinstance(bins, int) else (bins[1], bins[0])
            
            self.histograms[key] = {
                'tensor': torch.zeros((bins_y, bins_x), dtype=torch.long),
                'bins': (bins_x, bins_y),
                'range': settings['range']
            }
            
        self.corr = torch.zeros(1, 4, 4)

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

    def update(self, key, x_coords, y_coords):
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
        
        ax.set_xlabel(key.split('_')[0])
        ax.set_xlabel(key.split('_')[1])
        ax.set_title(f"Histogram for {key}")
        fig.colorbar(im, ax=ax, label='Prob. density')
        
        return fig, ax
        
    def forward(self, sample):
                
        # print('>> sample:', sample['2m_temperature'].shape)
                
        z_temperature = standardize(sample['2m_temperature']).flatten()
        z_u10m = standardize(sample['10m_u_component_of_wind']).flatten()
        z_v10m = standardize(sample['10m_v_component_of_wind']).flatten()
        z_mslp = standardize(sample['mean_sea_level_pressure']).flatten()
        
        data = torch.stack([
            z_temperature,
            z_u10m,
            z_v10m,
            z_mslp,
        ])
        
        self.update(
            't2m_u10m', 
            sample['2m_temperature'].flatten(), 
            sample['10m_u_component_of_wind'].flatten(),
        )
        
        self.update(
            't2m_v10m', 
            sample['2m_temperature'].flatten(), 
            sample['10m_v_component_of_wind'].flatten(),
        )
        
        self.update(
            't2m_mslp', 
            sample['2m_temperature'].flatten(), 
            sample['mean_sea_level_pressure'].flatten(),
        )
        
        self.update(
            'u10m_v10m', 
            sample['10m_u_component_of_wind'].flatten(), 
            sample['10m_v_component_of_wind'].flatten(),
        )
        
        self.update(
            'u10m_mslp', 
            sample['10m_u_component_of_wind'].flatten(), 
            sample['mean_sea_level_pressure'].flatten(),
        )
        
        self.update(
            'v10m_mslp', 
            sample['10m_v_component_of_wind'].flatten(), 
            sample['mean_sea_level_pressure'].flatten(),
        )
        
        corr = torch.cov(data).unsqueeze(0)   
        self.corr = torch.cat([self.corr, corr], dim=0)     
                
        return corr
import os

import torch
from torch import nn
from torch.nn import functional as F
        
import matplotlib.pyplot as plt
import matplotlib.colors as colors

class CorrelationMap(nn.Module):
    def __init__(self, grid):
        super().__init__()
        
        self.device = os.getenv('DEVICE')  # Get device from environment variable or default to CPU
        c = 4
        h = grid['lat'].shape[0]
        w = grid['lon'].shape[0]
        
        self.sum_c = torch.zeros(1, c, h, w, device=self.device, dtype=torch.float32)
        self.sum_c_sq = torch.zeros(1, c, h, w, device=self.device, dtype=torch.float32)
        self.sum_cc = torch.zeros(c, c, h, w, device=self.device, dtype=torch.float32)
        self.count = 0
        
    def forward(self, sample):
        
        data = torch.cat(
            [
                sample['2m_temperature'],
                sample['10m_u_component_of_wind'],
                sample['10m_v_component_of_wind'],
                sample['mean_sea_level_pressure'],
            ],
            dim=1,
        )
        
        self.count = self.count + 1
        
        self.sum_c = self.sum_c + data
        self.sum_cc = self.sum_cc + data*data.transpose(1, 0)
        self.sum_c_sq = self.sum_c_sq + data**2
        
        if self.count == 6480:
            self.visualize()
        
        return None
    
    def evaluate(self):
        
        sum_c_prod = self.sum_c * self.sum_c.transpose(1, 0)
        numerator = self.count * self.sum_cc - sum_c_prod
        
        var_term = self.count * self.sum_c_sq - self.sum_c_sq
        denominator_sq = var_term*var_term.transpose(1,0)
        denominator = torch.sqrt(denominator_sq + 1e-6)
        
        print(numerator.shape, denominator.shape)
        
        correlation_map = numerator / denominator
        
        return correlation_map
    
    def visualize(self):
        
        correlation_map = self.evaluate()
        
        print(correlation_map.shape)
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,6))
        
        im = ax.pcolormesh(correlation_map[1,2], vmin=-1, vmax=1, cmap='seismic')
        
        fig.colorbar(im, ax=ax)
        
        plt.savefig('/p/project1/hclimrep/vozar2/PACE-Toolkit/pace/plots/corrs_map.png')
        plt.close("all")
        
        return None
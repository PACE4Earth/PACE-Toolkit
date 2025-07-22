import torch
from torch import nn
from torch.nn import functional as F

from .operators import standardize

class SampleWiseCorrelation(nn.Module):
    def __init__(self, grid):
        super().__init__()
        
        # histogram creation
        # infer correlation tensor from grid?
        
    def forward(self, sample):
                
        print('>> sample:', sample['2m_temperature'].shape)
                
        # z = torch.zeros()
                
        z_temperature = standardize(sample['2m_temperature']).flatten()
        z_u10m = standardize(sample['10m_u_component_of_wind']).flatten()
        z_v10m = standardize(sample['10m_v_component_of_wind']).flatten()
        z_mslp = standardize(sample['mean_sea_level_pressure']).flatten()
        
        corr = (z_temperature * z_u10m).mean() - z_temperature.mean() * z_u10m.mean()
        
        print(corr.shape, corr)
        
        # histogram accumulation
        
        return corr
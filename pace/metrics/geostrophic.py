import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# import metpy

class GeostrophicWind(nn.Module):
    def __init__(self, grid, epsilon=1e-5):
        super().__init__()
        
                
        # Passed fields
        self.register_buffer("f", grid['f'])
        self.register_buffer("dx", grid['dx'])
        self.register_buffer("dy", grid['dy'])
        self.register_buffer("lat", grid['lat'])
        self.epsilon = epsilon

        # Sobel kernels
        kernel_dx = torch.tensor([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        kernel_dy = torch.tensor([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        
        self.register_buffer("kernel_dx", kernel_dx)
        self.register_buffer("kernel_dy", kernel_dy)

    def compute_geostrophic(self, sample):
        
        phi = sample['geopotential']
        
        # Ensure 4D shape [B, C, H, W]
        shape = phi.shape
        print('pre_pad', shape)
        while len(phi.shape) < 4:
            phi = phi.unsqueeze(0)

        # Padding before convolution (to preserve size)
        phi = F.pad(phi, (1, 1, 1, 1), mode='replicate')

        # Compute gradients using Sobel filters
        dphi_dx = F.conv2d(
            phi, 
            self.kernel_dx.repeat(shape[-3], 1, 1, 1),
            groups=shape[-3],
        ) / self.dx
        dphi_dy = F.conv2d(
            phi, 
            self.kernel_dy.repeat(shape[-3], 1, 1, 1),
            groups=shape[-3],
        ) / self.dy

        # Geostrophic wind equations
        u_g = dphi_dy / self.f
        v_g = dphi_dx / self.f

        # Apply latitude mask (only 30°–80° N/S)
        lat_mask = (self.lat >= 30) | (self.lat <= -30)
        lat_mask &= (self.lat <= 80) & (self.lat >= -80)  # Combine both ranges

        # Broadcast lat_mask to [1, 1, Y, 1]
        lat_mask_2d = lat_mask.view(1, 1, -1, 1)

        # Set outside of mask to NaN
        u_g = u_g.masked_fill(~lat_mask_2d, torch.nan)
        v_g = v_g.masked_fill(~lat_mask_2d, torch.nan)

        return u_g.view(shape), v_g.view(shape)        

    def forward(self, sample):
        
        """
        Computes ratio of ageostrophic and geostrophic wind component.

        Returns:
            ratio (tensor of shape [Y, X])
        """
        u_g, v_g = self.compute_geostrophic(sample)

        u_ag = sample['u_component_of_wind'] - u_g
        v_ag = sample['v_component_of_wind'] - v_g

        mag_geo = torch.sqrt(u_g**2 + v_g**2)
        mag_ageo = torch.sqrt(u_ag**2 + v_ag**2)

        ratio = mag_ageo / (mag_geo + self.epsilon)
        
        return ratio
                
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import metpy

class GeostrophicWind(nn.Module):
    def __init__(self, dx, dy, f):
        super().__init__()
        
        self.image_scale = 8.
        
        # Passed fields
        self.register_buffer("f", torch.tensor(f))
        self.register_buffer("dx", torch.tensor(dx))
        self.register_buffer("dy", torch.tensor(dy))

        # Sobel kernels
        kernel_dx = torch.tensor([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        kernel_dy = torch.tensor([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        
        self.register_buffer("kernel_dx", kernel_dx)
        self.register_buffer("kernel_dy", kernel_dy)


    def forward(self, phi):
        
        if type(phi)!=torch.Tensor:
            phi = torch.tensor(phi)
            
        shape = phi.shape
        while len(phi.shape)<4:
            phi = phi.unsqueeze(0)
        
        # this should be ammended for proper boundary conditions
        phi = F.pad(phi, (1, 1, 1, 1), mode='replicate')

        dphi_dx = F.conv2d(phi, self.kernel_dx) / self.dx
        dphi_dy = F.conv2d(phi, self.kernel_dy) / self.dy

        u_g =  dphi_dy / self.f
        v_g =  dphi_dx / self.f
        
        return u_g.view(shape), v_g.view(shape) # torch.cat((geopotential, u_g, v_g), dim=1)

from metpy import constants as mpconsts
from metpy.calc import (
    geospatial_gradient, 
    coriolis_parameter,
)

def lower_abs_boundary(xi):
    mask = np.abs(xi) < 1e-5
    xi[mask] = 1e-5 * np.sign(xi[mask]+1e-9)  

def metpy_geostrophic_wind(z, lat, lon):
    
    z = z.metpy.quantify()
    dx, dy = metpy.calc.lat_lon_grid_deltas(lon, lat)
    
    # f = coriolis_parameter(lat)
    # f.values = lower_abs_boundary(f.values)

    # dhdx, dhdy = geospatial_gradient(z, dx=dx, dy=dy)
    # u_g, v_g = -dhdy/f[:, None], dhdx/f[:, None]
    
    # return np.array(u_g), np.array(v_g)
    
    u_g, v_g = metpy.calc.geostrophic_wind(z, dx=dx, dy=dy)
    
    u_g = np.nan_to_num(u_g.values, nan=0)
    v_g = np.nan_to_num(v_g.values, nan=0)   

    return np.clip(u_g, -100, 100), np.clip(v_g, -100, 100)
    
# === Example usage ===
# ncfile = "your_file.nc"
# z, lat, lon = load_geopotential_height(ncfile, variable='z', level=500)
# u_g, v_g = compute_geostrophic_wind(z, lat, lon)
# ds_out = xr.Dataset({'u_g': u_g, 'v_g': v_g})
# print(ds_out)

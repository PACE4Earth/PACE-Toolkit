import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import (
    get_sobel_kernels,
    get_uniform_kernel,
    get_gaussian_kernel,
    pad_finite_difference,
)

class MassConservation(nn.Module):
    """
    Mass consistency metric (surface divergence)
    Computes tendency and global integral of surface pressure (derived from MSLP, geopotential, 2m temp) using barometric formula
    """
    def __init__(self, grid, smoothing='uniform', compute_divergence=True):
        super().__init__()
        self.g = 9.80665
        self.R = 287.05  # J/(kg·K)

        # Optionally, precompute area weights if grid info is available (to think about it????)
        if grid is not None and "area_weights" in grid:
            self.register_buffer("area_weights", grid["area_weights"])
        elif grid is not None and "lat" in grid:
            lat = grid["lat"]  # [H, W] or [H]
            if lat.ndim == 1:
                # Expand to 2D
                lat2d = lat.view(-1, 1).expand(-1, grid['dx'].shape[1])
            else:
                lat2d = lat
            weights = torch.cos(torch.deg2rad(lat2d))
            self.register_buffer("area_weights", weights)
        else:
            self.area_weights = None
     
    def forward(self, sample):
        """
        Inputs (sample dict):
            - 'mean_sea_level_pressure': MSLP [Pa]
            - 'temperature_2m': 2m temperature [K]
            - 'geopotential': geopotential [m^2/s^2]
            
        Returns:
            dict with 'surface_pressure' [Pa]
        """
        p_msl = sample["mean_sea_level_pressure"]
        T2m = sample["2m_temperature"]
        phi = sample["geopotential"]

        # Compute surface height
        h = phi / self.g

        # Barometric equation: ps = p_msl * exp(g*h/(R*T))
        ps = p_msl * torch.exp(self.g * h / (self.R * T2m))

        return {"surface_pressure": ps}
        
    def output_keys(self):
        return ["surface_pressure"]

    def evaluate(self, all_outputs, rank=0):
        if rank != 0:
            return

        # Stack along time
        ps_series = torch.stack([o["surface_pressure"] for o in all_outputs], dim=0)  # [time, B, H, W]
        dt_seconds = 6.0 * 3600.0
        ps_tendency = (ps_series[1:] - ps_series[:-1]) / dt_seconds  # [time-1, B, H, W]

        if hasattr(self, "area_weights") and self.area_weights is not None:
            weights = self.area_weights
        else:
            weights = torch.ones_like(ps_series[0, 0])

        global_mass_series = (ps_series * weights).sum(dim=(-2, -1))  # [time, B]

        return {
            "ps_series": ps_series,
            "ps_tendency": ps_tendency,
            "global_mass_series": global_mass_series
        }

"""
Notes:
------

def spatial_rmse(field):
    Compute RMSE over spatial dimensions (H, W) for each batch
    mse = torch.mean(field ** 2, dim=(-2, -1))
    rmse = torch.sqrt(mse)
    return rmse
"""    

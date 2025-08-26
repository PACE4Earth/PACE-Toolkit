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
    Primary focus: compute MSLP series and its global integral over time
    """
    def __init__(self, grid, smoothing='uniform', compute_divergence=True):
        super().__init__()

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
            - 'mean_sea_level_pressure': mean sea level pressure
            
        Returns:
            dict with 'mean_sea_level_pressure'
        """
        outputs = {}
        outputs["mean_sea_level_pressure"] = sample["mean_sea_level_pressure"]

        return outputs
        
    def output_keys(self):
        return ["mean_sea_level_pressure"]

    def evaluate(self, all_outputs, rank=0):
        """
        Aggregate over time:
        - compute MSLP series
        - compute time tendency (primary)
        - compute global integral of MSLP (mass conservation diagnostic)
        """
        if rank != 0:
            return

        # Stack along time
        mslp_series = torch.stack([o["mean_sea_level_pressure"] for o in all_outputs], dim=0)  # [time, B, H, W]
        dt_seconds = 6.0 * 3600.0
        mslp_tendency = (mslp_series[1:] - mslp_series[:-1]) / dt_seconds  # [time-1, B, H, W]

        # Compute global surface integral of MSLP (proxy for atmospheric mass)
        # use equal-area assumption or precomputed area weights
        if hasattr(self, "area_weights") and self.area_weights is not None:
            weights = self.area_weights  # [H, W]
        else:
            weights = torch.ones_like(mslp_series[0, 0])

        global_mass_series = (mslp_series * weights).sum(dim=(-2, -1))  # [time, B]

        return {
            "mslp_series": mslp_series,
            "mslp_tendency": mslp_tendency,
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

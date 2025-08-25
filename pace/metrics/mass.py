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
    Returns divergence field only; time tendency will be computed in postprocessing.
    """
    def __init__(self, grid, smoothing='uniform', compute_divergence=True):
        super().__init__()

        # Grid spacing and coordinates
        self.register_buffer("dx", grid['dx'])
        self.register_buffer("dy", grid['dy'])
        
        self.compute_divergence = compute_divergence

        # Sobel kernels for spatial gradients
        kernel_dx, kernel_dy = get_sobel_kernels()
        self.register_buffer("kernel_dx", kernel_dx)
        self.register_buffer("kernel_dy", kernel_dy)

        # Smoothing kernel
        if smoothing == 'uniform':
            self.pad = (1, 2, 1, 2)
            self.register_buffer('smoothing_kernel', get_uniform_kernel(kernel_size=4))
        elif smoothing == 'gaussian':
            self.pad = (4, 4, 4, 4)
            self.register_buffer('smoothing_kernel', get_gaussian_kernel(kernel_size=9, sigma=1.25))
        else:
            self.pad = (1, 2, 1, 2)
            self.register_buffer('smoothing_kernel', get_uniform_kernel(kernel_size=4))

    def output_keys(self):
        """
        Defines which variables will be stored in the final xarray.Dataset.
        """
        keys = ["surface_mass_divergence", "mean_sea_level_pressure"]
        return keys

    def compute_surface_divergence(self, rho, u, v):
        """
        Compute ∇·(ρ*v) at the surface.
        Inputs:
            rho: [B,1,H,W] air density
            u:   [B,1,H,W] u wind component
            v:   [B,1,H,W] v wind component
        Returns:
            div: [B,1,H,W] surface mass divergence
        """
        rho_u = rho * u
        rho_v = rho * v

        # Pad fields for finite diff
        rho_u = pad_finite_difference(rho_u, pad_width=(2, 2, 2, 2))
        rho_v = pad_finite_difference(rho_v, pad_width=(2, 2, 2, 2))
        
        # Divergence x
        div_x = F.conv2d(
            rho_u,
            self.kernel_dx.repeat(rho_u.shape[1], 1, 1, 1),
            groups=rho_u.shape[-3],
        )[..., 1:-1, 1:-1] / self.dx

        # Divergence y
        div_y = F.conv2d(
            rho_v,
            self.kernel_dy.repeat(rho_v.shape[1], 1, 1, 1),
            groups=rho_v.shape[-3],
        )[..., 1:-1, 1:-1] / self.dy

        return div_x + div_y
      
    def forward(self, sample):
        """
        Inputs (sample dict):
            - '2m_temperature': surface temperature
            - 'u_component_of_wind': u wind
            - 'v_component_of_wind': v wind
            - 'mean_sea_level_pressure': mean sea level pressure
            - optionally 'q': specific_humidity
        Returns:
            dict with 'surface_mass_divergence' + 'mean_sea_level_pressure'
        """
        outputs = {}

        if self.compute_divergence:
            u = sample["u_component_of_wind"]
            v = sample["v_component_of_wind"]

            # Density
            if "air_density" in sample:
                rho = sample["air_density"]
            else:
                R_d = 287.05  # J/(kg·K)
                T = sample["2m_temperature"][:,0:1,...]  # povrch
                qv = sample.get("specific_humidity", torch.zeros_like(T))
                p = sample["mean_sea_level_pressure"]
                rho = p / (R_d * T * (1 + 0.61*qv))

            outputs["surface_mass_divergence"] = self.compute_surface_divergence(rho, u, v) 

        outputs["mean_sea_level_pressure"] = sample["mean_sea_level_pressure"]

        return outputs
        
    def output_keys(self):
        return ["surface_mass_divergence", "mean_sea_level_pressure"]
"""
Notes:
------
- Time tendency ∂p/∂t for postproccessing

# def compute_pressure_tendency(sample_t, sample_t_minus, dt_hours=6.0):
#     dt_seconds = dt_hours * 3600.0
#     p_t = sample_t["mean_sea_level_pressure"]
#     p_tm = sample_t_minus["mean_sea_level_pressure"]
#     tendency = (p_t - p_tm) / dt_seconds
#     return tendency

# def evaluate_mass_consistency(divergence, pressure_tendency):
#     g = 9.80665
#     residual = pressure_tendency + g * divergence
#     l2_score = torch.sqrt((residual**2).mean())
#     return residual, l2_score
"""    

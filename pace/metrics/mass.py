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
    def __init__(self, grid, epsilon=1e-5, smoothing='uniform', compute_divergence=False):
        super().__init__()

        # Grid spacing and coordinates
        self.register_buffer("dx", grid['dx'])
        self.register_buffer("dy", grid['dy'])
        self.register_buffer("lat", grid['lat'])
        self.epsilon = epsilon
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

    def compute_surface_divergence(self, rho, u, v):
        """Compute ∇·(ρ v) at the surface."""
        rho_u = rho * u
        rho_v = rho * v

        # Pad fields
        rho_u = pad_finite_difference(rho_u, pad_width=(2, 2, 2, 2))
        rho_v = pad_finite_difference(rho_v, pad_width=(2, 2, 2, 2))

        div_x = F.conv2d(
            rho_u,
            self.kernel_dx.repeat(rho_u.shape[-3], 1, 1, 1),
            groups=rho_u.shape[-3],
        )[..., 1:-1, 1:-1] / self.dx

        div_y = F.conv2d(
            rho_v,
            self.kernel_dy.repeat(rho_v.shape[-3], 1, 1, 1),
            groups=rho_v.shape[-3],
        )[..., 1:-1, 1:-1] / self.dy

        return div_x + div_y
      
    def forward(self, sample):
        outputs = {}
        outputs["surface_pressure"] = sample.get("mean_sea_level_pressure")

        if self.compute_divergence:
            u = sample["u_component_of_wind"]
            v = sample["v_component_of_wind"]

            # vypocet hustoty, ak nie je v datasete
            if "air_density" in sample:
                rho = sample["air_density"]
            else:
                R_d = 287.05  # J/(kg·K)
                T = sample["temperature"][:,0:1,...]  # povrchová hladina
                qv = sample.get("specific_humidity", torch.zeros_like(T))
                p = outputs["surface_pressure"]
                rho = p / (R_d * T * (1 + 0.61 * qv))

            outputs["surface_mass_divergence"] = self.compute_surface_divergence(rho, u, v)

        return outputs

    def output_keys(self):
        keys = ['mean_sea_level_pressure']
        if self.compute_divergence:
            keys += ['surface_mass_divergence']
        return keys

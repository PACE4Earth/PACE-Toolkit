import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operators import (
    get_sobel_kernels,
    get_uniform_kernel,
    get_gaussian_kernel,
    pad_finite_difference,
)


class GeostrophicWind(nn.Module):
    """
    Metric computing the ratio of ageostrophic to geostrophic wind.

    Args:
        grid (dict): Dictionary containing grid metadata and constants:
            - f (Tensor): Coriolis parameter [1/s]
            - dx (Tensor): Grid spacing in x-direction [m]
            - dy (Tensor): Grid spacing in y-direction [m]
            - lat (Tensor): Latitude array [degrees]
        epsilon (float, optional): Small value to avoid division by zero.
                                   Defaults to 1e-5.
        smoothing (str, optional): Smoothing kernel type for final ratio field.
                                   One of {"uniform", "gaussian"}.
                                   Defaults to "uniform".

    Inputs:
        sample (dict): Dictionary with fields:
            - "geopotential" (Tensor): Geopotential field [B, L, H, W]
            - "u_component_of_wind" (Tensor): U-wind [B, L, H, W]
            - "v_component_of_wind" (Tensor): V-wind [B, L, H, W]

    Outputs:
        Tensor: Smoothed ratio field [B, L, H, W]
    """

    def __init__(self, grid, epsilon=1e-5, smoothing="uniform"):
        super().__init__()

        # Store grid parameters as non-trainable buffers
        self.register_buffer("f", grid["f"])
        self.register_buffer("dx", grid["dx"])
        self.register_buffer("dy", grid["dy"])
        self.register_buffer("lat", grid["lat"])
        self.epsilon = epsilon

        # Sobel filters for finite-difference gradient approximation
        kernel_dx, kernel_dy = get_sobel_kernels()
        self.register_buffer("kernel_dx", kernel_dx)
        self.register_buffer("kernel_dy", kernel_dy)

        # Define smoothing kernel applied to final ratio
        # (hard-coded kernel sizes chosen for 0.25° → 2° resolution scaling)
        if smoothing == "uniform":
            self.pad = (1, 2, 1, 2)
            self.register_buffer("smoothing_kernel", get_uniform_kernel(kernel_size=4))
        elif smoothing == "gaussian":
            self.pad = (4, 4, 4, 4)
            self.register_buffer(
                "smoothing_kernel", get_gaussian_kernel(kernel_size=9, sigma=1.25)
            )
        else:
            raise ValueError(f"Unknown smoothing method '{smoothing}'")

    def compute_geostrophic(self, sample):
        """
        Compute geostrophic wind components from geopotential.

        Args:
            sample (dict): Must contain "geopotential" [B, L, H, W].

        Returns:
            tuple[Tensor, Tensor]: (u_g, v_g) geostrophic wind components
                                   with same shape as input geopotential.
        """
        phi = sample["geopotential"]

        # Ensure 4D shape [B, L, H, W] for conv2d compatibility
        shape = phi.shape
        while len(phi.shape) < 4:
            phi = phi.unsqueeze(0)

        # Pad field before convolution to allow centered finite differences
        phi = pad_finite_difference(phi, pad_width=(2, 2, 2, 2))

        # Gradient dphi/dx
        dphi_dx = F.conv2d(
            phi,
            self.kernel_dx.repeat(shape[-3], 1, 1, 1),
            groups=shape[-3],
        )[..., 1:-1, 1:-1] / self.dx

        # Gradient dphi/dy
        dphi_dy = F.conv2d(
            phi,
            self.kernel_dy.repeat(shape[-3], 1, 1, 1),
            groups=shape[-3],
        )[..., 1:-1, 1:-1] / self.dy

        # Geostrophic balance equations
        u_g = dphi_dy / self.f
        v_g = dphi_dx / self.f

        # Apply latitude mask: only valid in midlatitudes (30°–80° N/S)
        lat_mask = (self.lat >= 30) | (self.lat <= -30)
        lat_mask &= (self.lat <= 80) & (self.lat >= -80)
        lat_mask_2d = lat_mask.view(1, 1, -1, 1)  # broadcastable mask

        u_g = u_g.masked_fill(~lat_mask_2d, torch.nan)
        v_g = v_g.masked_fill(~lat_mask_2d, torch.nan)

        return u_g.view(shape), v_g.view(shape)

    def forward(self, sample):
        """
        Compute the ratio of ageostrophic to geostrophic wind.

        Args:
            sample (dict): Must contain "geopotential", "u_component_of_wind",
                           and "v_component_of_wind".

        Returns:
            Tensor: Smoothed ratio field [B, L, H, W]
        """
        u_g, v_g = self.compute_geostrophic(sample)

        # Ageostrophic component = model wind - geostrophic wind
        u_ag = sample["u_component_of_wind"] - u_g
        v_ag = sample["v_component_of_wind"] - v_g

        mag_geo = torch.sqrt(u_g**2 + v_g**2)
        mag_ageo = torch.sqrt(u_ag**2 + v_ag**2)

        ratio = mag_ageo / (mag_geo + self.epsilon)

        # Apply spatial smoothing to reduce grid-scale noise
        kernel = self.smoothing_kernel.repeat(ratio.shape[-3], 1, 1, 1).to(ratio.dtype)
        ratio = F.pad(ratio, self.pad, "replicate")
        ratio = F.conv2d(ratio, kernel, groups=ratio.shape[-3], padding=0)

        return ratio

    def output_keys(self):
        """
        Define output key(s) produced by this metric.

        Returns:
            list[str]: ["geostrophic_wind_ratio"]
        """
        return ["geostrophic_wind_ratio"]

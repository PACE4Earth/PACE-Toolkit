import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GeostrophicWind(nn.Module):
    def __init__(self, dx, dy, f, lat):
        super().__init__()

        # Register constants as buffers (will be part of model, but not parameters)
        self.register_buffer("f", torch.tensor(f))
        self.register_buffer("dx", torch.tensor(dx))
        self.register_buffer("dy", torch.tensor(dy))
        self.register_buffer("lat", torch.tensor(lat))  # Shape: [Y]

        # Sobel kernels for spatial gradients
        kernel_dx = torch.tensor([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        kernel_dy = torch.tensor([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0

        self.register_buffer("kernel_dx", kernel_dx)
        self.register_buffer("kernel_dy", kernel_dy)

    def forward(self, phi):
        # Ensure 4D shape [B, C, H, W]
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi)
        shape = phi.shape
        while len(phi.shape) < 4:
            phi = phi.unsqueeze(0)

        # Padding before convolution (to preserve size)
        phi = F.pad(phi, (1, 1, 1, 1), mode='replicate')

        # Compute gradients using Sobel filters
        dphi_dx = F.conv2d(phi, self.kernel_dx) / self.dx
        dphi_dy = F.conv2d(phi, self.kernel_dy) / self.dy

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

    def compute_with_actual_wind(self, phi, u, v, eps=1e-5):
        """
        Compute geostrophic, ageostrophic components, and ageo/geo ratio.

        Returns:
            u_g, v_g, u_ag, v_ag, ratio (all tensors of shape [Y, X])
        """
        u_g, v_g = self.forward(phi)

        u_ag = u - u_g
        v_ag = v - v_g

        mag_geo = torch.sqrt(u_g**2 + v_g**2)
        mag_ageo = torch.sqrt(u_ag**2 + v_ag**2)

        ratio = mag_ageo / (mag_geo + eps)
        return u_g, v_g, u_ag, v_ag, ratio
        
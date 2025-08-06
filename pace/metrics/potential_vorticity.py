import torch
import torch.nn as nn
import torch.nn.functional as F

class PotentialVorticity(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.Rd = 287.05
        self.cp = 1004.0
        self.g = 9.80665
        self.epsilon = 1e-5

        # Grid fields
        self.p_levels = grid['pressure_levels'].float() * 100.0  # hPa → Pa
        self.dx = grid['dx'].float()  # [H, W]
        self.dy = grid['dy'].float()  # [H, W]
        self.f = grid['f'].float()    # [H, W]

        # Determine if global domain (wraparound in longitude)
        lon = grid['lon'].float()
        lon_span = lon[-1] - lon[0]
        self.is_global = lon_span > 359.99

    def compute_potential_temperature(self, T, p):
        """
        θ = T * (p0/p)^(R/cp)
        """
        p0 = 1e5  # 1000 hPa
        return T * (p0 / p.view(1, -1, 1, 1)) ** (self.Rd / self.cp)

    def compute_vertical_gradient(self, theta):
        """
        Central ∂θ/∂p using explicit padding with NaN.
        Input: [B, L, H, W] → Output: [B, L, H, W]
        """
        dp = self.p_levels.to(theta.device)  # [L]

        # Pad levels with NaNs on both sides
        theta_pad = F.pad(theta, (0, 0, 0, 0, 1, 1), value=float('nan'))  # pad level dim
        dp_pad = F.pad(dp, (1, 1), value=float('nan'))

        # Compute central difference on padded data
        dtheta = theta_pad[:, 2:] - theta_pad[:, :-2]  # [B, L, H, W]
        dp_mid = dp_pad[2:] - dp_pad[:-2]              # [L]

        dtheta_dp = dtheta / dp_mid.view(1, -1, 1, 1)
        return dtheta_dp

    def compute_relative_vorticity(self, u, v):
        """
        ζ = dv/dx - du/dy with global wraparound in longitude if needed.
        Output: [B, L, H, W]
        """
        dx = self.dx.to(u.device)  # [H, W]
        dy = self.dy.to(u.device)  # [H, W]

        if self.is_global:
            # Use rolling for periodic wraparound in longitude (W dim)
            dv_dx = (torch.roll(v, shifts=-1, dims=-1) - torch.roll(v, shifts=1, dims=-1)) / (
                torch.roll(dx, shifts=-1, dims=-1) + torch.roll(dx, shifts=1, dims=-1))
        else:
            v_pad = F.pad(v, (1, 1, 0, 0), value=float('nan'))
            dx_pad = F.pad(dx, (1, 1), value=float('nan'))
            dv_dx = (v_pad[..., :, 2:] - v_pad[..., :, :-2]) / (dx_pad[:, 2:] + dx_pad[:, :-2])

        # Latitude (non-periodic)
        u_pad = F.pad(u, (0, 0, 1, 1), value=float('nan'))
        dy_pad = F.pad(dy, (0, 0, 1, 1), value=float('nan'))
        du_dy = (u_pad[..., 2:, :] - u_pad[..., :-2, :]) / (dy_pad[2:, :] + dy_pad[:-2, :])

        return dv_dx - du_dy

    def forward(self, sample):
        """
        Compute PV in PVU: [B, L, H, W]
        """
        T = sample['temperature']   # [B, L, H, W]
        u = sample['u_component_of_wind']
        v = sample['v_component_of_wind']

        # Ensure levels are ordered top to bottom
        p_levels = self.p_levels.to(T.device)
        if p_levels[0] > p_levels[-1]:
            T = torch.flip(T, dims=[1])
            u = torch.flip(u, dims=[1])
            v = torch.flip(v, dims=[1])
            p_levels = torch.flip(p_levels, dims=[0])

        # Compute θ
        theta = self.compute_potential_temperature(T, p_levels)  # [B, L, H, W]

        # Vertical derivative ∂θ/∂p
        dtheta_dp = self.compute_vertical_gradient(theta)  # [B, L, H, W]

        # Relative vorticity ζ
        zeta = self.compute_relative_vorticity(u, v)  # [B, L, H, W]

        # Total vorticity η = f + ζ
        f = self.f.to(T.device).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        eta = f + zeta  # [B, L, H, W]

        # PV = -g * η * ∂θ/∂p
        pv = -self.g * eta * dtheta_dp  # [B, L, H, W]

        # Convert to PVU
        pv_pvu = pv * 1e6  # [B, L, H, W]
        return pv_pvu

    def output_keys(self):
        return ['potential_vorticity']

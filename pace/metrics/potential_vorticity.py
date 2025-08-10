import torch
import torch.nn as nn

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
        p0 = 1e5  # 1000 hPa in Pa
        return T * (p0 / p.view(1, -1, 1, 1)) ** (self.Rd / self.cp)

    def compute_vertical_gradient(self, theta):
        """
        Compute ∂θ/∂p with:
          - forward difference at top level
          - backward difference at bottom level
          - central difference inside
        
        Input: [B, L, H, W]
        Output: [B, L, H, W]
        """
        dp = self.p_levels.to(theta.device)  # [L], pressure levels in Pa, increasing downward

        B, L, H, W = theta.shape
        dtheta_dp = torch.empty_like(theta)

        # Forward difference at top (level 0)
        dtheta_dp[:, 0] = (theta[:, 1] - theta[:, 0]) / (dp[1] - dp[0])

        # Backward difference at bottom (level L-1)
        dtheta_dp[:, -1] = (theta[:, -1] - theta[:, -2]) / (dp[-1] - dp[-2])

        # Central difference in interior levels
        # numerator: theta[i+1] - theta[i-1]
        # denominator: dp[i+1] - dp[i-1]
        dtheta = theta[:, 2:] - theta[:, :-2]           # [B, L-2, H, W]
        dp_mid = dp[2:] - dp[:-2]                        # [L-2]
        dtheta_dp[:, 1:-1] = dtheta / dp_mid.view(1, -1, 1, 1)

        return dtheta_dp

    def compute_relative_vorticity(self, u, v):
        """
        ζ = dv/dx - du/dy with global wraparound in longitude if needed.
        Output: [B, L, H, W]
        """
        dx = self.dx.to(u.device)  # [H, W]
        dy = self.dy.to(u.device)  # [H, W]

        B, L, H, W = u.shape

        # Compute dv/dx
        if self.is_global:
            # Periodic in longitude (W)
            dv_dx = (torch.roll(v, shifts=-1, dims=-1) - torch.roll(v, shifts=1, dims=-1)) / (
                torch.roll(dx, shifts=-1, dims=-1) + torch.roll(dx, shifts=1, dims=-1))
        else:
            # Non-periodic: forward/backward differences at edges, central inside
            dv_dx = torch.empty_like(v)
            # Forward difference at left edge (W=0)
            dv_dx[..., :, :, 0] = (v[..., :, :, 1] - v[..., :, :, 0]) / (dx[:, 1] + dx[:, 0])
            # Backward difference at right edge (W=-1)
            dv_dx[..., :, :, -1] = (v[..., :, :, -1] - v[..., :, :, -2]) / (dx[:, -1] + dx[:, -2])
            # Central difference inside
            dv_dx[..., :, :, 1:-1] = (v[..., :, :, 2:] - v[..., :, :, :-2]) / (dx[:, 2:] + dx[:, :-2])

        # Compute du/dy (latitude is non-periodic)
        du_dy = torch.empty_like(u)
        # Forward difference at top edge (H=0)
        du_dy[..., :, 0, :] = (u[..., :, 1, :] - u[..., :, 0, :]) / (dy[1, :] + dy[0, :])
        # Backward difference at bottom edge (H=-1)
        du_dy[..., :, -1, :] = (u[..., :, -1, :] - u[..., :, -2, :]) / (dy[-1, :] + dy[-2, :])
        # Central difference inside
        du_dy[..., :, 1:-1, :] = (u[..., :, 2:, :] - u[..., :, :-2, :]) / (dy[2:, :] + dy[:-2, :])

        return dv_dx - du_dy

    def forward(self, sample):
        """
        Compute PV in PVU: [B, L, H, W]

        Set top and bottom PV levels to NaN to avoid unreliable boundary values.
        """
        T = sample['temperature']       # [B, L, H, W]
        u = sample['u_component_of_wind']
        v = sample['v_component_of_wind']

        # Ensure levels are ordered top to bottom (pressure increasing downward)
        p_levels = self.p_levels.to(T.device)
        if p_levels[0] > p_levels[-1]:
            T = torch.flip(T, dims=[1])
            u = torch.flip(u, dims=[1])
            v = torch.flip(v, dims=[1])
            p_levels = torch.flip(p_levels, dims=[0])

        # Compute θ
        theta = self.compute_potential_temperature(T, p_levels)  # [B, L, H, W]

        # Vertical gradient ∂θ/∂p
        dtheta_dp = self.compute_vertical_gradient(theta)         # [B, L, H, W]

        # Relative vorticity ζ
        zeta = self.compute_relative_vorticity(u, v)              # [B, L, H, W]

        # Total vorticity η = f + ζ
        f = self.f.to(T.device).unsqueeze(0).unsqueeze(0)         # [1, 1, H, W]
        eta = f + zeta                                             # [B, L, H, W]

        # PV = -g * η * ∂θ/∂p
        pv = -self.g * eta * dtheta_dp                             # [B, L, H, W]

        # Convert to PVU
        pv_pvu = pv * 1e6                                          # [B, L, H, W]

        # Mask top and bottom vertical levels as NaN (unreliable)
        pv_pvu[:, 0] = float('nan')
        pv_pvu[:, -1] = float('nan')

        return pv_pvu

    def output_keys(self):
        return ['potential_vorticity']

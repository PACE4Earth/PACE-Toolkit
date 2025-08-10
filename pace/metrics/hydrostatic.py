import torch
import torch.nn as nn

class HydrostaticBalance(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.Rd = 287.05  # J/(kg·K), gas constant for dry air
        self.epsilon = 1e-5
        p_levels = grid.get('pressure_levels', None)
        self.p_levels = p_levels.float() * 100.0  # Convert hPa to Pa if needed

    def virtual_temperature(self, temperature, specific_humidity):
        """
        Compute virtual temperature.
        Tv = T * (1 + 0.61 * q)
        """
        return temperature * (1 + 0.61 * specific_humidity)

    def compute_dphi_hydrostatic(self, Tv, p_levels):
        """
        Compute hydrostatic geopotential differences using the discrete form:
        dPhi = -Rd * Tv * dlnp

        Args:
            Tv: Virtual temperature [B, L, H, W]
            p_levels: Pressure levels [L] in Pa (must be increasing from top to bottom)

        Returns:
            dphi_hydro: Hydrostatic geopotential differences [B, L-1, H, W]
        """
        ln_p = torch.log(p_levels)  # [L]
        dlnp = ln_p[1:] - ln_p[:-1]  # [L-1]

        # Mean virtual temperature between adjacent levels
        Tv_mean = 0.5 * (Tv[:, :-1] + Tv[:, 1:])  # [B, L-1, H, W]

        # Reshape dlnp for broadcasting
        dlnp = dlnp.view(1, -1, 1, 1)  # [1, L-1, 1, 1]

        # Hydrostatic geopotential difference
        dphi_hydro = -self.Rd * Tv_mean * dlnp  # [B, L-1, H, W]

        return dphi_hydro

    def forward(self, sample):
        """
        Compute deviation from hydrostatic balance.

        Input sample dict:
            'geopotential': [B, L, H, W] in m²/s²
            'temperature': [B, L, H, W] in K
            'specific_humidity': [B, L, H, W] in kg/kg
        'pressure_levels': list or array of [L] in hPa (top to bottom) - from the grid

        Returns:
            abs_error_padded: Absolute deviation [B, L, H, W]
            rel_error_padded: Relative deviation [0–1], unitless
        """
        phi = sample['geopotential']          # [B, L, H, W]
        T = sample['temperature']             # [B, L, H, W]
        q = sample['specific_humidity']       # [B, L, H, W]

        # Convert pressure levels to tensor 
        p_levels = self.p_levels.to(phi.device)

        # Flip pressure levels if in descending order
        if p_levels[0] > p_levels[-1]:
            p_levels = torch.flip(p_levels, dims=[0])
            phi = torch.flip(phi, dims=[1])
            T = torch.flip(T, dims=[1])
            q = torch.flip(q, dims=[1])

        # Compute virtual temperature
        Tv = self.virtual_temperature(T, q)  # [B, L, H, W]

        # Compute hydrostatic geopotential thickness
        dphi_hydro = self.compute_dphi_hydrostatic(Tv, p_levels)  # [B, L-1, H, W]

        # Actual geopotential difference between levels
        dphi_actual = phi[:, 1:] - phi[:, :-1]  # [B, L-1, H, W]

        # Absolute and relative error (per level, height, width)
        abs_error = torch.abs(dphi_actual - dphi_hydro)  # [B, L-1, H, W]
        rel_error = abs_error / (torch.abs(dphi_hydro) + self.epsilon)  # [B, L-1, H, W]

        # RMSE aggregated over levels only, keep H and W
        mse = ((dphi_actual - dphi_hydro) ** 2).mean(dim=1)  # mean over L-1, shape [B, H, W]
        rmse = torch.sqrt(mse)  # [B, H, W]

        # Add a singleton dimension for levels to get [B, 1, H, W]
        rmse = rmse.unsqueeze(1)  # [B, 1, H, W]

        # Pad abs_error and rel_error to match original level count (L)
        nan_pad = torch.full_like(abs_error[:, :1], float('nan'))  # [B, 1, H, W]
        abs_error_padded = torch.cat([nan_pad, abs_error], dim=1)  # [B, L, H, W]
        rel_error_padded = torch.cat([nan_pad, rel_error], dim=1)  # [B, L, H, W]

        # Return errors and rmse map
        return abs_error_padded, rel_error_padded, rmse

    
    def output_keys(self):
        return ['hydrostatic_abs_error', 'hydrostatic_rel_error', 'hydrostatic_rmse']

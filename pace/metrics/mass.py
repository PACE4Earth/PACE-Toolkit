import torch
import torch.nn as nn

class MassConservation(nn.Module):
    """
    Mass consistency metric for ML weather model outputs on a pressure-based vertical grid.

    Reconstructs approximate surface pressure from MSLP using
    temperature and geopotential, then computes total atmospheric mass.
    """

    def __init__(self, grid):
        super().__init__()
        self.g = 9.80665   # gravity [m/s^2]
        self.R = 287.05    # gas constant for dry air [J/kg/K]

        # Compute area weights from grid spacing
        dx = grid["dx"]  # [H, W] in meters
        dy = grid["dy"]  # [H, W] in meters
        area = dx * dy    # [H, W]
        self.register_buffer("area", area)

        # Identify lowest model level (highest pressure)
        p_levels = grid["pressure_levels"]
        self.lowest_idx = torch.argmax(p_levels)  # index of highest pressure

    def forward(self, sample):
        """
        Compute total atmospheric mass in kg from reconstructed surface pressure.

        Parameters
        ----------
        sample : dict
            - 'mean_sea_level_pressure': [B, H, W], in Pa
            - 'temperature': [B, L, H, W], in K
            - 'geopotential': [B, L, H, W], in m^2/s^2
        
        Returns
        -------
        torch.tensor
            - 'total_mass': [B], total atmospheric mass in kg
        """
        mslp = sample["mean_sea_level_pressure"]  # [B,H,W]

        # Get surface geopotential and temperature at lowest level
        phi_s = sample["geopotential"][:, self.lowest_idx, :, :]  # [B,H,W], m^2/s^2
        T_s = sample["temperature"][:, self.lowest_idx, :, :]     # [B,H,W], K

        # Reconstruct surface pressure from MSLP
        ps = mslp * torch.exp(phi_s / (self.R * T_s))  # [B,H,W]

        # Area-weighted sum to get total mass
        weights = self.area.unsqueeze(0)  # [1,H,W]
        total_mass = (ps * weights).sum(dim=(-2,-1)) / self.g  # [B]

        return total_mass

    def output_keys(self):
        return ["total_mass"]

    # def evaluate(self, all_outputs, dt_seconds=6*3600, rank=0):
    #     """
    #     Evaluate mass consistency across multiple time steps.

    #     Parameters
    #     ----------
    #     all_outputs : list of dict
    #         Output of `forward(sample)` for each time step.
    #     dt_seconds : float
    #         Time difference between consecutive samples in seconds.
    #     rank : int
    #         Only rank 0 computes postprocessing in distributed mode.
        
    #     Returns
    #     -------
    #     dict
    #         - 'global_mass_series': [time, B]
    #         - 'global_mass_tendency': [time-1, B], temporal derivative
    #     """
    #     if rank != 0:
    #         return

    #     # Stack global mass across time
    #     global_mass_series = torch.stack([o["global_mass"] for o in all_outputs], dim=0)  # [time, B]

    #     # Temporal derivative
    #     global_mass_tendency = (global_mass_series[1:] - global_mass_series[:-1]) / dt_seconds  # [time-1, B]

    #     return {
    #         "global_mass_series": global_mass_series,
    #         "global_mass_tendency": global_mass_tendency
    #     }

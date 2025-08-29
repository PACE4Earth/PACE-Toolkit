import torch
import torch.nn as nn

class MassConservation(nn.Module):
    """
    Mass consistency metric for ML weather model outputs on a pressure-based vertical grid.

    Computes the total atmospheric mass (dry + moisture) by
    integrating mean sea-level pressure (MSLP) over the grid.
    """

    def __init__(self, grid):
        super().__init__()
        self.g = 9.80665  # gravity [m/s^2]

        # Compute area weights from grid spacing
        dx = grid["dx"]  # [H, W] in meters
        dy = grid["dy"]  # [H, W] in meters
        area = dx * dy                  # [H, W]
        self.register_buffer("area", area)  # [H, W]

    def forward(self, sample):
        """
        Compute total atmospheric mass in kg from MSLP.

        Parameters
        ----------
        sample : dict
            - 'mslp': [B, H, W], mean sea-level pressure in Pa
        
        Returns
        -------
        dict
            - 'global_mass': [B], total atmospheric mass in kg
        """
        mslp = sample["mean_sea_level_pressure"]  # [B, H, W]

        # Area-weighted sum, divide by gravity to get mass
        weights = self.area.unsqueeze(0)  # [1, H, W] for batch broadcasting
        total_mass = (mslp * weights).sum(dim=(-2, -1)) / self.g  # [B, 1]

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

import torch
import torch.nn as nn

class HumidityConsistency(nn.Module):
    def __init__(self, grid):
        super().__init__()
        self.e0 = 611.2       # Pa
        self.T0 = 273.15      # K
        self.Lv = 2.5e6       # J/kg
        self.Rv = 461.5       # J/(kg·K)
        self.eps = 0.622
        self.epsilon = 1e-8
        self.p_levels = grid.get('pressure_levels', None).float() * 100.0  # hPa → Pa

    def forward(self, sample):
        """
        Compute RH (%) = (q / qs) * 100
        Inputs:
            T [B, L, H, W] (K)
            q [B, L, H, W] (kg/kg)
        Returns:
            relative_humidity [B, L, H, W] (%)
        """
        T = sample['temperature']
        q = sample['specific_humidity']
        p = self.p_levels.to(T.device).view(1, -1, 1, 1)

        # es(T) from Clausius–Clapeyron [Pa]
        es = self.e0 * torch.exp((self.Lv / self.Rv) * ((1.0 / self.T0) - (1.0 / (T + self.epsilon))))

        # qs(T,p) = (eps * es) / (p - (1 - eps) * es)
        qs = (self.eps * es) / (p - (1 - self.eps) * es + self.epsilon)

        # RH in %
        rh = (q / (qs + self.epsilon)) * 100.0
        return rh

    def output_keys(self):
        return ['relative_humidity']

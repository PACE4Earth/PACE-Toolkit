import torch
import torch.nn as nn

#Energz conservations modelu metric for pace
#2025-08-14     MP      First attempt   
#2025-08-18     MP      Smoothing code  

class EnergyConservation(nn.Module):
    """
    Compute column-integrated total energy (internal + kinetic + latent).
    """
    def __init__(self, grid):
        super().__init__()
        self.cvd = 718.0  # J/(kg·K), specific heat for dry air (approx)
        self.cvv = 1410.0 # J/(kg·K), water vapor (constant volume)
        self.L_v = 2.5e6  # J/kg, latent heat of vaporization     

        #In case if we want do this inline or in postprocessing
        self.g = 9.81     # m/s²

        self.p_levels = grid.get('pressure_levels', None)

        if self.p_levels is not None:
            self.p_levels = self.p_levels.float() * 100.0  # hPa → Pa

    def compute_total_energy(self, T, u, v, q):
        """
        T: [B, L, H, W] in K
        u, v: [B, L, H, W] in m/s
        q: [B, L, H, W] specific humidity in kg/kg
        """
        c_v_moist = (1 - q) * self.cvd + q * self.cvv

        E_int = c_v_moist * T

        E_kin = 0.5 * (u**2 + v**2)

        E_lat = self.L_v * q

        return E_int + E_kin + E_lat  # [B, L, H, W]

    def vertical_integrate(self, E_total):
        """  
        Integrate over pressure levels using simple layer thickness dp/g
        """
        dp = self.p_levels[1:] - self.p_levels[:-1]  # [L-1]
        dp = dp.view(1, -1, 1, 1)  # broadcast to [1, L-1, 1, 1]
        E_layer = 0.5 * (E_total[:, :-1] + E_total[:, 1:]) * dp / self.g
        return E_layer.sum(dim=1)  # [B, H, W]

    def forward(self, sample):
        """
        Input:
            'temperature': T [B,L,H,W]
            'u': u wind [B,L,H,W]
            'v': v wind [B,L,H,W]
            'q': specific humidity [B,L,H,W]
        
        Outputs:
            dict with energy fields (total energy or total energy per column)
        """
        T = sample['temperature']
        u = sample['u']
        v = sample['v']
        q = sample['q']

        # Compute total energy
        E_total = self.compute_total_energy(T, u, v, q)

        outputs = {'total_energy': E_total}

        # Column-integrated
        if self.p_levels is not None:
            outputs['total_energy_column'] = self.vertical_integrate(E_total)

        return outputs

    #to postprocessing : temporal tendency and RMSE
    ## Column-integrated
    #    E_column = self.vertical_integrate(E_total)  # [B,H,W]
    #
    #    # Temporal tendency if 'time' dimension exists
    #    if 'time' in sample:
    #        dt = sample['time'][1] - sample['time'][0]  # assume scalar
    #        dE_dt = (E_column[1:] - E_column[:-1]) / dt
    #    else:
    #        dE_dt = torch.zeros_like(E_column)
    #
    #    # RMSE over B,H,W
    #    rmse = torch.sqrt((dE_dt**2).mean(dim=(0,1,2), keepdim=True))  # scalar [1,1,1]
    #
    #    return E_column, dE_dt, rmse

    def output_keys(self):
        keys = ['total_energy']
        if self.p_levels is not None:
            keys.append('total_energy_column')
        return keys

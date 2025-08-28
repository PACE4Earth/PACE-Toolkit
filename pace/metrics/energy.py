import torch
import torch.nn as nn


class EnergyConservation(nn.Module):
    """
    Compute column-integrated total energy (internal + kinetic + latent).
    """
    def __init__(self, grid):
        super().__init__()
        self.cpd = 1005.0  # J/(kg·K), cp dry air
        self.cpv = 1952.0  # J/(kg·K), cp water vapor
        self.L_v = 2.5e6  # J/kg, latent heat of vaporization     
        self.g = 9.81     # m/s²

        self.p_levels = grid.get('pressure_levels', None)

        if self.p_levels is not None:
            self.p_levels = self.p_levels.float() * 100.0

    def compute_total_energy(self, T, u, v, q, z):
        """
        T: [B, L, H, W] in K
        u, v: [B, L, H, W] in m/s
        q: [B, L, H, W] specific humidity in kg/kg
        z: [B,L,H,W] geopotential (m^2/s^2)
        """
        c_moist = (1 - q) * self.cpd + q * self.cpv

        E_int = c_moist * T #entalpy

        E_kin = 0.5 * (u**2 + v**2)

        E_lat = self.L_v * q

        E_pot = z

        return E_int + E_kin + E_lat + E_pot  # [B, L, H, W]

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
            'geopotential': [B,L,H,W]
        
        Outputs:
            dict with energy fields (total energy or total energy per column)
        """
        T = sample['temperature']                       
        u = sample['u_component_of_wind']               
        v = sample['v_component_of_wind']               
        q = sample['specific_humidity'] 
        z = sample['geopotential']

        # Compute total energy
        E_total = self.compute_total_energy(T, u, v, q, z)
        outputs = {'total_energy': E_total}

        # Column-integrated
        if self.p_levels is not None:
            outputs['total_energy_column'] = self.vertical_integrate(E_total)

        return outputs

    def output_keys(self):
        keys = ['total_energy']
        if self.p_levels is not None:
            keys.append('total_energy_column')
        return keys

"""
Notes:
------
- To postprocessing : temporal tendency and RMSE

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
"""    

import os

### more than one too many for the test version
# it might be necessary to use xarray for i/o backends,
# unless done from scratch using the backends directly

import numpy as np
import xarray as xr
import torch

import matplotlib.pyplot as plt

from metrics.geostrophic import (
    GeostrophicWind, 
    metpy_geostrophic_wind
)

class NetCDFDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        
        ### files and fields indexing
        
        files = [
            os.path.join(folder_path, f) 
            for f in os.listdir(folder_path) 
            if f.endswith('.nc')
        ]
        files.sort()
        if not files:
            raise ValueError(f"No .nc files found in {folder_path}")
        
        # might be handled differently, either in batched
        self.index_map = []
        for file in files:
            with xr.open_dataset(file, engine='netcdf4') as ds:
                for valid_time in ds['valid_time']:
                    self.index_map.append(
                        (file, valid_time.values)
                    )      
        
        ### static fields preparation
        
        # assuming computation on a single grid within one dataset
        # might need to add more filtering
        with xr.open_dataset(
            self.index_map[0][0], 
            engine='netcdf4'
        ).sel(valid_time=self.index_map[0][1]) as ds:
            self.lats = ds['latitude'].values
            self.lons = ds['longitude'].values    
        
        # can be cast to torch.tensor(np.ndarray, device=DEVICE)
        
        # self.y = self.lats * 111.32e3
        # self.x = self.lons * np.cos(np.deg2rad(self.lats)) * 111.32e3
        
        # one-linear gradient, dimension expansion and value scaling
        self.cos_lat = np.cos(np.deg2rad(self.lats))
        self.dy = (np.gradient(self.lats))[:, None] * np.ones_like(self.lons)[None, :] * 111.32e3
        
        # coriolis
        omega = 7.2921e-5
        self.f = 2 * omega * np.sin(np.deg2rad(self.lats))[:, None] * np.ones_like(self.lons)[None, :]

        mask = np.abs(self.f) < 1e-5
        self.f[mask] = 1e-5 * np.sign(self.f[mask]+1e-9)    
    
        ### ugly one linear gradient on periodic boundary
        self.dx = (np.gradient(
            np.concatenate(
                [self.lons[[-1]]-360., self.lons, self.lons[[0]]+360.], 0
            )
        )[1:-1])[None, :] * np.cos(np.deg2rad(self.lats))[:, None] * 111.32e3
        
        if ('z' in list(ds.variables)) and ('u' in list(ds.variables)) and ('v' in list(ds.variables)):
            self.geostrophy = GeostrophicWind(self.dx, self.dy, self.f)
        else:
            self.geostrophy = torch.nn.Identity() # this can be a handy one-liner
        
    def __len__(self):
        return len(self.index_map)

# can be extented or modified. if we have dx, dy, f masks, we can cast
# to torch tensor, and load the masks by dataset.f, reducing memory
# footprint due to coordinates duplication. this can be done with xarray
# as well, but that is just barebone numpy and nobody does that

    def __getitem__(self, idx):
        file_path = self.index_map[idx][0]
        valid_time = self.index_map[idx][1]
        
        with xr.open_dataset(file_path, engine='netcdf4') as ds:
            return {
                'x': ds.load().sel(valid_time=valid_time), 
                't': valid_time,
            }

    
if __name__=="__main__":
    
    dataset = NetCDFDataset(os.path.join(os.getcwd(), 'data'))
    
    if not os.path.exists('../__tmp_outs'):
        try:
            os.mkdir('../__tmp_outs')
        except:
            print('Wanted to plot, but cannot make a directory ../__tmp_outs')
    
    for it in range(dataset.__len__()):
        
        sample_dict = dataset[it]
        tau, valid_time = sample_dict['x'], sample_dict['t']
                
        for level in tau['pressure_level']:
            phi_xr = tau.sel(pressure_level=level)['z']
            phi = tau.sel(pressure_level=level)['z'].values
            u = tau.sel(pressure_level=level)['u'].values
            v = tau.sel(pressure_level=level)['v'].values
            # print(valid_time.values, level.values, phi.shape)
            
            # compute by torch
            u_gt, v_gt = dataset.geostrophy.forward(phi)
            
            # compute by metpy
            u_gx, v_gx = metpy_geostrophic_wind(phi_xr, dataset.lats, dataset.lons)
            
            # print(torch.nn.functional.l1_loss(u_gt, torch.tensor(u_gx)).item())        
            
            ### for v_g a dirty trick was necessary (prob. f mask?)
            ### also i saved some type checking, so youll get a UserWarning
            ### about recasting tensor to tensor
            
            for (u_g, v_g, backend) in [
                (u_gt, v_gt, 'torch'),
                (u_gx, v_gx, 'xarray')
            ]:
                print(
                    f'{backend:>6} L1(u,u_g) [ms-1]: ',
                    torch.nn.functional.l1_loss(
                        torch.tensor(u),
                        torch.tensor(u_g).clamp(2*u.min(), 2*u.max()),
                        reduction='mean',
                    ).item(),
                    f'{backend:>6} L1(v,v_g) [ms-1]: ',
                    torch.nn.functional.l1_loss(
                        torch.tensor(v),
                        torch.tensor(v_g).clamp(2*v.min(), 2*v.max()),
                        reduction='mean',
                    ).item(),
                )
            
            # print(phi.shape, u_g.shape, v_g.shape)
            
                if os.path.exists('../__tmp_outs'):
                    
                    delta_u = (np.array(u**2 + v**2)**.5) - (np.array(u_g**2 + v_g**2))**.5
                    # delta_u = (u**2+v**2)**.5 - np.array((u_g**2+v_g**2)**.5)
                    
                    # _ = torch.nn.functional.mse_loss(
                    #     torch.tensor(u**2+v**2)**.5,
                    #     torch.tensor(u_g**2+v_g**2)**.5,
                    #     reduction='none',
                    # ).numpy()
                        
                    
                    plt.title(
                        r'$||u||_2 - ||u_g||_2$' +
                        # f'\n RMSE = {np.mean(delta_u):1.2f}' $ needs
                        f'\n{backend} {int(level.values)}hPa {str(valid_time.astype('datetime64[h]')).replace('-', '').replace('T', '')}'
                    )
                    plt.contour(
                        dataset.lons,
                        dataset.lats, 
                        phi,
                        colors='k',
                        levels=20,
                        linewidths=1,
                    )
                    plt.pcolormesh(
                        dataset.lons,
                        dataset.lats, 
                        # v-np.array(v_g),
                        # (u**2+v**2)**.5 - np.array((u_g**2+v_g**2)**.5),
                        delta_u,
                        vmin = -20,
                        vmax = 20,
                        cmap = 'seismic',
                    )
                    
                    plt.tight_layout()
                    
                    plt.colorbar()
                    
                    plt.savefig(f'../__tmp_outs/{backend}_{int(level.values)}_{str(valid_time.astype('datetime64[h]')).replace('-', '').replace('T', '')}.png')
                    plt.close('all')
    
    # tau = dataset[0]
    # if dataset.geostrophy != None:
    #     for valid_time in tau['valid_time']:
    #         rho = tau.sel(valid_time=valid_time)
    
    # print(tau)
    
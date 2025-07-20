import os
import json

import numpy as np
import torch
import xarray
import netCDF4

from pathlib import Path

def inspect_nc(path):
    with xarray.open_dataset(path) as ds:
        print(ds)   
            
def check_required_fields(path, metric_requirements, aliases):

    ret = {}
    
    with xarray.open_dataset(path, engine='netcdf4') as ds:
        available_fields = [
            var_name
            for var_name, _ in ds.variables.items()
        ]
        
    for metric in metric_requirements:
        ret[metric] = None
        try:
            for alias in aliases[metric]:
                if alias in available_fields:
                    ret[metric] = alias
                    break
            else:
                ret[metric] = None
        except Exception as e:
            # print(e)
            ret[metric] = None
    
    return ret

def get_grid(path):
    with xarray.open_dataset(path, engine='netcdf4') as ds:
        try:
            lats = ds['latitude'].values
        except KeyError:
            lats = ds['lat'].values
        except Exception as e:
            print(e)
            return None
        
        try:
            lons = ds['longitude'].values
        except KeyError:
            lons = ds['lon'].values
        except Exception as e:
            print(e)
            return None    
    
    cos_lat = np.cos(np.deg2rad(lats))
    dy = (np.gradient(lats))[:, None] * np.ones_like(lons)[None, :] * 111.32e3
    
    # coriolis
    omega = 7.2921e-5
    f = 2 * omega * np.sin(np.deg2rad(lats))[:, None] * np.ones_like(lons)[None, :]

    mask = np.abs(f) < 1e-5
    f[mask] = 1e-5 * np.sign(f[mask]+1e-9)    

    ### ugly one linear gradient on periodic boundary
    dx = (np.gradient(
        np.concatenate(
            [lons[[-1]]-360., lons, lons[[0]]+360.], 0
        )
    )[1:-1])[None, :] * np.cos(np.deg2rad(lats))[:, None] * 111.32e3
    
    return {
        'lon': torch.tensor(lons),
        'lat': torch.tensor(lats),
        'dx': torch.tensor(dx),
        'dy': torch.tensor(dy),
        'f': torch.tensor(f)
    }
            
class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, config_path=None):

        # load aliases
        aliases_path = Path(__file__).resolve().parent.parent / "configs" / "aliases.json"
        with open(aliases_path, "r") as f:
            self.aliases = json.load(f)
        
        # Load JSON config
        config_path = Path(__file__).resolve().parent.parent / "configs" / "graphcast_extended.json"
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Expand env variables in the config
        def expand_env_vars(obj):
            if isinstance(obj, dict):
                return {k: expand_env_vars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [expand_env_vars(i) for i in obj]
            elif isinstance(obj, str):
                return os.path.expandvars(obj)
            else:
                return obj

        config = expand_env_vars(config)

        # Set dataset config values as attributes
        dataset_config = config["dataset"]
        for k, v in dataset_config.items():
            setattr(self, k, v)
            
        # check files
        print('Preparing files...', end=' ')
        self.files = [
            os.path.join(self.path, f, 'output.nc') 
            if not os.path.isfile(os.path.join(self.path, f)) 
            else os.path.join(self.path, f)
            for f in os.listdir(self.path)
        ]
        self.files.sort()
        
        if dataset_config['name'] == "graphcast_extended":
            self.lead_times = 40
        else:
            self.lead_times = 1
        
        print('Done\n')
        
        print('Static fields setup...', end=' ')
        self.grid = get_grid(self.files[0])
        print('Done\n')
            
        # inspect_nc(self.files[0])
            
        # check required fields for the metrics
        print('Checking required field for metrics...')
        metrics_path = Path(__file__).resolve().parent.parent / "configs" / "variables_for_metrics.json"
        with open(metrics_path, 'r') as f:
            metrics_requirements = json.load(f)
        
        self.metrics = {}
        for metric in config['metrics']:
            req_for_this_metric = metrics_requirements[metric]
            found_for_this_metric = check_required_fields(
                self.files[0], 
                req_for_this_metric,
                self.aliases
            )
            if any(f==None for f in found_for_this_metric.values()):
                print(f'{metric:<23}', 'is missing field/s for computation.')
            else:
                print(f'{metric:<23}', 'is complete.')
                self.metrics[metric] = found_for_this_metric            
        
        names_in_files = []
        canonical_names = []
        for k,v in self.metrics.items():
            for canonical, true in v.items():
                canonical_names.append(canonical)
                names_in_files.append(true)
        self.canonical_names = list(dict.fromkeys(canonical_names))
        self.requested_names = list(dict.fromkeys(names_in_files))
        
        print('Done\n\nFields to be loaded:')
        [
            print(f'{cn:<20} <- {rn}')
            for (cn, rn) in zip(self.canonical_names, self.requested_names)
        ]
        
    def __len__(self):
        return len(self.files)*self.lead_times

    def __getitem__(self, idx):
        
        file_idx = idx // self.lead_times
        time_idx = idx % self.lead_times

        # 2. Get the corresponding file path
        file_path = self.files[file_idx]
        
        fields = {}
        
        with xarray.open_dataset(file_path, engine='netcdf4') as ds:
            base_time = ds['time'].values[0]
            ds = ds.isel(time=time_idx)
            valid_time = ds['time'].values
            for rq, cn in zip(self.requested_names, self.canonical_names):
                tau = torch.tensor(ds[rq].values)
                tau = tau.squeeze()
                if tau.dim()<3:
                    tau = tau.unsqueeze(0)
                fields[cn] = tau
                print('\t', cn, tau.shape)
        
        lead_time = (valid_time - base_time) / np.timedelta64(1, 'h')
        base_time = base_time.astype('datetime64[s]').astype(np.int64)
        valid_time = valid_time.astype('datetime64[s]').astype(np.int64)
        # back conversion performed via:
        # sample['valid_time'].astype('datetime64[s]')
        # print(base_time, lead_time, valid_time)
        
        fields['base_time'] = base_time
        fields['valid_time'] = valid_time
        fields['lead_time'] = lead_time
        
        return fields
        
import os
import json

import torch
import xarray
import netCDF4

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
    
    # print(available_fields)   
    
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
            print(e)
            ret[metric] = None
               
    # for k,v in ret.items():
    #     print(k,v)
    
    return ret
            
class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, config_path):
        
        # load aliases
        with open("./pace/configs/aliases.json", "r") as f:
            self.aliases = json.load(f)
        
        # load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        for k, v in config['dataset'].items():
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
        print('Done')
        
        # inspect_nc(self.files[0])
            
        # check required fields for the metrics
        print('Checking required field for metrics...')
        with open(self.metrics_path, 'r') as f:
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
        
        print('Done\nFields to be loaded:')
        [
            print(f'{cn:<20} <- {rn}')
            for (cn, rn) in zip(self.canonical_names, self.requested_names)
        ]
            
    def __len__(self):
        return len(self.files)*len(self.lead_times)

    def __getitem__(self, idx):
        
        fields = {}
        
        with xarray.open_dataset(self.files[idx], engine='netcdf4') as ds:
            for rq, cn in zip(self.requested_names, self.canonical_names):
                tau = torch.tensor(ds[rq].values)
                if tau.dim()<5:
                    tau = tau.unsqueeze(2)
                fields[cn] = tau
        
        return fields

if __name__=="__main__":
    
    config_path = './pace/configs/graphcast_extended.json'
    # config_path = './pace/configs/test_data.json'
    
    dataset = UnifiedDataset(config_path=config_path)
    
    sample = dataset[0]
    for rq, field in sample.items():
        print(rq, field.shape)
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
    )
        
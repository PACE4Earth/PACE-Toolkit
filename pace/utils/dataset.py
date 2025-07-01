import os
import json

import torch
import xarray
import netCDF4

def inspect_nc(path):
    with xarray.open_dataset(path) as ds:
        print(ds)
            
            
def check_required_fields(path, metric_requirements):
    
    with xarray.open_dataset(path) as ds:
        available_fields = [
            var_name
            for var_name, var in ds.variables.items()
        ]
        
    ok = all(
        req in available_fields
        for req in metric_requirements
    )
    
    return ok
    
            
class UnifiedDataset(torch.utils.data.Dataset):
    def __init__(self, config_path):
        
        # load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        for k, v in config['dataset'].items():
            setattr(self, k, v)
            
        # check files
        print('Preparing files...', end=' ')
        self.files = [
            os.path.join(self.path, timestamp_dir, 'output.nc')
            for timestamp_dir in os.listdir(self.path)
        ]
        self.files.sort()
        print('Done')
        
        # inspect_nc(self.files[0])
            
        # check required fields for the metrics
        print('Checking required field for metrics...', end=' ')
        with open(self.metrics_path, 'r') as f:
            metrics_requirements = json.load(f)
        
        req_fields = []
        for metric in config['metrics']:
            req_for_this_metric = metrics_requirements[metric]
            if check_required_fields(
                self.files[0], 
                metrics_requirements[metric],
            ): req_fields.extend(req_for_this_metric)
            
        self.required_fields = list(set(req_fields))
        
        print('Done\nFields to be loaded:\n', self.required_fields)
            
    def __len__(self):
        return len(self.files)*len(self.lead_times)

    def __getitem__(self, idx):
        
        fields = []
        
        with xarray.open_dataset(self.files[idx]) as ds:
            for var in self.required_fields:
                # print(var, end=' ')
                tau = torch.tensor(ds[var].values[[0]])
                if tau.dim()<5:
                    tau = tau.unsqueeze(2)
                # print(tau.shape)
                fields.append(tau)
        
        fields = torch.cat(fields, dim=2)
        
        # print(fields.shape)
        
        return fields

if __name__=="__main__":
    
    config_path = "/p/project1/hclimrep/vozar2/PACE-Toolkit/pace/configs/graphcast_extended.json"
    
    dataset = UnifiedDataset(config_path=config_path)
    
    _ = dataset[0]
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
    )
        
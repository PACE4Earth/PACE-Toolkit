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
    
    with xarray.open_dataset(path) as ds:
        available_fields = [
            var_name
            for var_name, var in ds.variables.items()
        ]
       
    for metric in metric_requirements:
        ret[metric] = None
        try:
            for alias in aliases[metric]:
                if alias in available_fields:
                    ret[metric] = alias
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
            
        # print(self.path)
        
        # check files
        print('Preparing files...', end=' ')
        self.files = [
            os.path.join(self.path, f, 'output.nc') if not os.path.isfile else os.path.join(self.path, f)
            for f in os.listdir(self.path)
        ]
        self.files.sort()
        print('Done')
        
        # inspect_nc(self.files[0])
            
        # check required fields for the metrics
        print('Checking required field for metrics...', end=' ')
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
                print(metric, ' is missing field for computation.')
            else:
                self.metrics[metric] = found_for_this_metric            
        
        tmp = []
        for k,v in self.metrics.items():
            for canonical, true in v.items():
                tmp.append(true)
        self.required_fields = list(set(tmp))
        
        print('Done\nFields to be loaded:')
        [
            print(req)
            for req in self.required_fields 
        ]
            
    def __len__(self):
        return len(self.files)*len(self.lead_times)

    def __getitem__(self, idx):
        
        fields = {}
        
        with xarray.open_dataset(self.files[idx]) as ds:
            for var in self.required_fields:
                # print(var, end=' ')
                tau = torch.tensor(ds[var].values[[0]])
                if tau.dim()<5:
                    tau = tau.unsqueeze(2)
                fields[var] = tau
        
        return fields

if __name__=="__main__":
    
    # config_path = "/p/project1/hclimrep/vozar2/PACE-Toolkit/pace/configs/graphcast_extended.json"
    config_path = './pace/configs/test_data.json'
    
    dataset = UnifiedDataset(config_path=config_path)
    
    sample = dataset[0]
    for var, field in sample.items():
        print(var, field.shape)
        
    dataloader = torch.utils.data.DataLoader(
        dataset,
    )
        
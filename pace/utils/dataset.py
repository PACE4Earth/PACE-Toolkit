import torch
import xarray

class Dataset(torch.utils.data.Dataset):
    def __init__(self, cfg):
        
        ...
        
    def __getitem__(self, idx):
        
        ...
        
        return
    
def get_dataloader(cfg):
    
    dataset = Dataset(cfg)
    
    dataloader = torch.utils.data.dataloader(
        dataset, 
        cfg,
    )
    
    return dataloader
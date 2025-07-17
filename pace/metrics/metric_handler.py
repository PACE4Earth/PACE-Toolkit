import torch
from torch import nn

from geostrophic_new import GeostrophicWind

METRIC_MODULES = {
    'geostrophic_balance' : GeostrophicWind
} 

class MetricHandler(nn.Module):
    def __init__(self, grid, metrics: list[str]):
        super().__init__()
                
        self.metrics = {metric: METRIC_MODULES[metric](grid) for metric in metrics}

    def forward(self, sample):
        
        outputs = {}
        
        for metric, module in self.metrics:
            outputs[metric] = module(sample)
        
        return outputs
    
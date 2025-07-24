import torch
from torch import nn

from .geostrophic import GeostrophicWind
from .correlation import SampleWiseCorrelation
from .hydrostatic import HydrostaticBalance 

METRIC_MODULES = {
    'geostrophic_balance' : GeostrophicWind,
    'correlation' : SampleWiseCorrelation,
    'hydrostatic_balance': HydrostaticBalance
} 

class MetricHandler(nn.Module):
    def __init__(self, grid, metrics: list[str]):
        super().__init__()
        self.metrics = {metric: METRIC_MODULES[metric](grid) for metric in metrics}

    def forward(self, sample):
        outputs = {}

        for metric_name, module in self.metrics.items():
            result = module(sample)

            # If hydrostatic_balance returns tuple of two tensors:
            if metric_name == 'hydrostatic_balance':
                abs_error, rel_error = result
                outputs[f'{metric_name}_abs_error'] = abs_error
                outputs[f'{metric_name}_rel_error'] = rel_error
            else:
                # For other metrics, assume single tensor output
                outputs[metric_name] = result

        return outputs

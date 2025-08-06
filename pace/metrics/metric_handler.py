import torch
from torch import nn

from .geostrophic import GeostrophicWind
from .correlation import SampleWiseCorrelation
from .hydrostatic import HydrostaticBalance 
from .correlation_map import CorrelationMap
from .humidity import HumidityConsistency

METRIC_MODULES = {
    'geostrophic_balance': GeostrophicWind,
    'correlation': SampleWiseCorrelation,
    'correlation_map': CorrelationMap,
    'hydrostatic_balance': HydrostaticBalance,
    'humidity_temperature': HumidityConsistency,
}

class MetricHandler(nn.Module):
    def __init__(self, grid, metrics: list[str]):
        super().__init__()
        self.metrics = {
            metric_name: METRIC_MODULES[metric_name](grid)
            for metric_name in metrics
        }

    def forward(self, sample: dict) -> dict:
        """
        Compute all registered metrics on the given sample.
        Returns a dict of outputs with descriptive names.
        """
        outputs = {}

        for metric_name, module in self.metrics.items():
            result = module(sample)

            # Handle multiple outputs (e.g., tuple or dict)
            if isinstance(result, tuple):
                keys = module.output_keys() if hasattr(module, 'output_keys') else [f"{metric_name}_{i}" for i in range(len(result))]
                for k, val in zip(keys, result):
                    outputs[k] = val
            elif isinstance(result, dict):
                for k, v in result.items():
                    outputs[f"{metric_name}_{k}"] = v
            else:
                if hasattr(module, 'output_keys'):
                    key = module.output_keys()[0]
                else:
                    key = metric_name
                outputs[key] = result


        return outputs

    def get_metric_names(self) -> list:
        """Returns a flat list of all expected output keys from all metrics."""
        names = []
        for metric_name, module in self.metrics.items():
            if hasattr(module, 'output_keys'):
                names.extend(module.output_keys())
            else:
                names.append(metric_name)
        return names

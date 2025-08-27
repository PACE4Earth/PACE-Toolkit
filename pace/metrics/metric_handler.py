import os
import json
from pathlib import Path

import torch
from torch import nn

from .geostrophic import GeostrophicWind
from .correlation import GenericHistogram
from .correlation_map import CorrelationMap
from .hydrostatic import HydrostaticBalance
from .humidity import HumidityConsistency
from .potential_vorticity import PotentialVorticity
from .mass import MassConservation
from .energy import EnergyConservation

# Registry of available metric modules. 
# Keys must match those expected in the config file.
METRIC_MODULES = {
    'geostrophic_balance': GeostrophicWind,
    'correlation': GenericHistogram,
    'correlation_map': CorrelationMap,
    'correlation_corrdiff': GenericHistogram,
    'correlation_map_corrdiff': CorrelationMap,
    'hydrostatic_balance': HydrostaticBalance,
    'humidity_temperature': HumidityConsistency,
    'potential_vorticity': PotentialVorticity,
    'mass_conservation': MassConservation,
    'energy_conservation': EnergyConservation,
}

def move_dict_to_device(tensor_dict, device=None):
    """
    Move all tensors in a dictionary to the specified device.

    Args:
        tensor_dict (dict): Mapping of keys to values. Values may be torch.Tensors
                           or other objects (which are left untouched).
        device (torch.device | None): Target device. If None, defaults to
                                      'cuda' if available else 'cpu'.

    Returns:
        dict: Copy of tensor_dict with tensors moved to the target device.
    """
    
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return {
        key: value.to(device) if ((isinstance(value, torch.Tensor)) and value.device != device) else value
        for key, value in tensor_dict.items()
    }

class MetricHandler(nn.Module):
    """
    A handler class for managing and executing multiple physical consistency metrics.

    Each metric is defined as a module (nn.Module) with a standard interface:
      - It must be callable on a sample dict.
      - It should provide an `output_keys()` method that defines the names of
        its outputs in order.

    The MetricHandler:
      * Loads a configuration specifying which metrics to run and which outputs to keep.
      * Initializes and stores metric modules.
      * On `forward()`, executes selected metrics and returns a dictionary of outputs.

    Example config (JSON):
    {
        "metrics": {
            "geostrophic_balance": ["geostrophic_wind_ratio"],
            "hydrostatic_balance": ["hydrostatic_rmse"],
            "humidity_temperature": ["all"]   // "all" selects all available outputs
        }
    }

    Args:
        grid: Grid information passed to metric modules (domain-specific).
        config_path (str | Path): Path to JSON configuration file.
        metrics (list[str]): List of metric names to activate from the config.
    """

    def __init__(self, grid, config_path, metrics: list[str],):
        super().__init__()
        
        with open(config_path, "r") as f:
            config = json.load(f)

        if "metrics" not in config or not isinstance(config["metrics"], dict):
            raise ValueError(f"Invalid config format in {config_path}: missing 'metrics' dict")
            
        self.metrics_config = {}      # metric_name -> selected output keys
        self.metrics = {}             # metric_name -> metric module
        self.available_keys_map = {}  # metric_name -> all available output keys
        requested_metrics = set(metrics)

        for metric_name, keys in config["metrics"].items():
            # Only build requested metrics
            if metric_name not in requested_metrics:
                continue 

            if metric_name not in METRIC_MODULES:
                raise KeyError(f"Unknown metric '{metric_name}' in config")

            # Initialize module if implemented
            try:
                module = METRIC_MODULES[metric_name](grid)
                module.to(os.getenv('DEVICE'))
            except Exception as e:
                print(e)
                module = lambda tau: 0

            # Get all outputs in their defined order
            available_keys = (
                module.output_keys()
                if hasattr(module, "output_keys")
                else [metric_name]
            )

            # Handle "all" case, otherwise filter by available_keys order
            if keys and len(keys) == 1 and str(keys[0]).lower() == "all":
                keys = available_keys
            else:
                # Filter keys in the order of available_keys, not config order
                keys = [k for k in available_keys if k in keys]
                missing = [k for k in keys if k not in available_keys]
                if missing:
                    raise KeyError(
                        f"Invalid output key(s) {missing} for metric '{metric_name}'. "
                        f"Available: {available_keys}"
                    )

            self.metrics_config[metric_name] = keys
            self.metrics[metric_name] = module
            self.available_keys_map[metric_name] = available_keys
            
    def forward(self, sample: dict) -> dict:
        """
        Run all active metrics on a given sample.

        Args:
            sample (dict): A data sample containing tensors and metadata.

        Returns:
            dict: Mapping of selected metric output keys to results.
                  Always includes 'idx' if present in sample.
        """
        outputs = {}
        sample = move_dict_to_device(sample, os.getenv('DEVICE'))

        outputs['idx'] = sample.get('idx', torch.tensor(0, device=os.getenv('DEVICE')))

        for metric_name, module in self.metrics.items():
            result = module(sample)
            available_keys = self.available_keys_map[metric_name]
            selected_keys = self.metrics_config[metric_name]

            if isinstance(result, tuple):
                if len(result) != len(available_keys):
                    raise ValueError(
                        f"Metric '{metric_name}' returned {len(result)} outputs, "
                        f"but output_keys() reports {len(available_keys)}."
                    )

                # Map by index according to output_keys
                for idx, key in enumerate(available_keys):
                    if key in selected_keys:
                        outputs[key] = result[idx]

            elif isinstance(result, dict):
                for key in available_keys:
                    if key in selected_keys:
                        if key not in result:
                            raise KeyError(
                                f"Key '{key}' from output_keys not found in dict output of metric '{metric_name}'"
                            )
                        outputs[key] = result[key]

            else:
                # Single-output case
                if len(available_keys) != 1:
                    raise ValueError(
                        f"Metric '{metric_name}' returned a single output but "
                        f"output_keys() has {len(available_keys)}."
                    )
                if available_keys[0] in selected_keys:
                    outputs[available_keys[0]] = result
                    
        return outputs

    def get_metric_names(self) -> list[str]:
        """
        Returns all output keys in the order of each metric's output_keys(),
        filtered according to config.
        """
        ordered_keys = []
        for metric_name, available_keys in self.available_keys_map.items():
            selected_keys = self.metrics_config[metric_name]
            ordered_keys.extend([k for k in available_keys if k in selected_keys])
        return ordered_keys

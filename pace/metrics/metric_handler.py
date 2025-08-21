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
    'case_studies': None,
}

def move_dict_to_device(tensor_dict, device=None):
    """
    Moves all tensors in a dictionary to a specified device.

    Args:
        tensor_dict (dict): A dictionary where values can be torch.Tensors.
        device (torch.device): The target device to move tensors to.

    Returns:
        dict: A new dictionary with all tensors moved to the target device.
    """
    
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    return {
        key: value.to(device) if ((isinstance(value, torch.Tensor)) and value.device != device) else value
        for key, value in tensor_dict.items()
    }

class MetricHandler(nn.Module):
    def __init__(self, grid, config_path, metrics: list[str],):
        """
        config_path: path to JSON config file, with format:
        {
            "metrics": {
                "geostrophic_balance": ["geostrophic_wind_ratio"],
                "hydrostatic_balance": ["hydrostatic_rmse"],
                "humidity_temperature": ["all"]   # "all" = all available outputs
            }
        }
        """
        super().__init__()
        
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # config_path = Path(__file__).resolve().parent.parent / "configs" / "config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        if "metrics" not in config or not isinstance(config["metrics"], dict):
            raise ValueError(f"Invalid config format in {config_path}: missing 'metrics' dict")
            
        self.metrics_config = {}
        self.metrics = {}
        self.available_keys_map = {}  # metric_name -> ordered output_keys
        requested_metrics = set(metrics)

        for metric_name, keys in config["metrics"].items():
            if metric_name not in requested_metrics:
                continue  # skip metrics not requested

            if metric_name not in METRIC_MODULES:
                raise KeyError(f"Unknown metric '{metric_name}' in config")

            try:
                module = METRIC_MODULES[metric_name](grid)
                module.to(os.getenv('DEVICE'))
            except:
                module = lambda tau: 0

            # Get authoritative key order from module, or fallback
            if hasattr(module, "output_keys"):
                available_keys = module.output_keys()
            else:
                available_keys = [metric_name]

            # Handle "all" case (case-insensitive)
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
            
        # print(self.metrics.keys())

    def forward(self, sample: dict) -> dict:
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

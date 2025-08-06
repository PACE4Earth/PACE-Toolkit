import os
import shutil

import zarr
import numpy as np
import numcodecs

import torch
import torch.nn as nn

class IndexedZarrSaver(nn.Module):
    """
    A torch.nn.Module that saves forecasts and provides integer-based access. 🗂️

    This module saves data to a hierarchical structure: `root/base_time/lead_time/`.
    It also maintains an internal `_index` array within the Zarr store, mapping
    an integer index to each (base_time, lead_time) pair.

    This allows the saver instance to be used like a dataset for easy retrieval:
    `data = saver[0]`
    """
    def __init__(self, path: str, mode: str = 'a'):
        super().__init__()
        path = os.path.join(path, 'test_out.zarr')
        self.path = path
        self.root = zarr.open_group(self.path, mode=mode)

        # Initialize or load the index array that maps integers to forecast paths
        try:
            self.index_array = self.root['_index']
        except KeyError:
            # Create the index array if it doesn't exist
            self.index_array = self.root.create_dataset(
                '_index',
                shape=(0, 2), # Two columns for [base_time, lead_time]
                chunks=(1024, 2),
                dtype=object,
                object_codec=numcodecs.JSON() # Handles variable-length strings
            )
        print(f"IndexedZarrSaver initialized at '{path}'. Found {len(self)} existing forecasts.")

    def __len__(self) -> int:
        """Returns the total number of saved forecasts."""
        return len(self.index_array)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a forecast by its integer index.

        Args:
            idx (int): The integer index of the forecast to retrieve.

        Returns:
            dict: A dictionary containing the 'outputs' as numpy arrays
                  and 'attrs' from the Zarr group.
        """
        if idx >= len(self):
            raise IndexError("Index out of range")

        base_time_key, lead_time_key = self.index_array[idx]
        group = self.root[base_time_key][lead_time_key]

        # Load all arrays from the group into a dictionary
        outputs = {name: arr[:] for name, arr in group.arrays()}
        
        return {
            "base_time": base_time_key,
            "lead_time": lead_time_key,
            "outputs": outputs,
            "attrs": dict(group.attrs)
        }

    def forward(self, sample: dict) -> dict:
        """Saves forecast data and updates the internal index."""
        base_times = sample['base_time']
        lead_times = sample['lead_time']
        
        sample.pop('lead_time')
        sample.pop('base_time')
        
        is_batch = isinstance(base_times, (list, tuple))
        new_indices = []

        if is_batch:
            paths_to_append = []
            for i in range(len(base_times)):
                base_t_key = str(base_times[i])
                lead_t_key = str(lead_times[i])
                paths_to_append.append([base_t_key, lead_t_key])

                lead_group = self.root.require_group(base_t_key).require_group(lead_t_key)

                for name, tensor in sample.items():
                    if tensor != None:
                        lead_group[name] = tensor[i].detach().cpu().numpy()
            
            # Efficiently append all new paths to the index array at once
            self.index_array.append(paths_to_append)
        else:
            base_t_key = str(base_times)
            lead_t_key = str(lead_times)

            lead_group = self.root.require_group(base_t_key).require_group(lead_t_key)

            for name, tensor in sample.items():
                if tensor != None:
                    lead_group[name] = tensor.detach().cpu().numpy()
                    print('saved')
            
            # Append the new path to the index array
            self.index_array.append([[base_t_key, lead_t_key]])

        return sample
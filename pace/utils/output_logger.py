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
        self.path = path  # e.g., outputs/graphcast.zarr

        # Automatically remove existing directory if it exists
        if os.path.exists(self.path):
            print(f"[IndexedZarrSaver] Removing existing zarr directory: {self.path}")
            shutil.rmtree(self.path)

        self.root = zarr.open_group(self.path, mode='w')

        # Create the index array
        self.index_array = self.root.create_dataset(
            '_index',
            shape=(0, 2),
            chunks=(1024, 2),
            dtype=object,
            object_codec=numcodecs.JSON()
        )

        print(f"IndexedZarrSaver initialized at '{self.path}'. Ready to save forecasts.")

    def __len__(self) -> int:
        return len(self.index_array)

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self):
            raise IndexError("Index out of range")

        base_time_key, lead_time_key = self.index_array[idx]
        group = self.root[base_time_key][lead_time_key]

        outputs = {name: arr[:] for name, arr in group.arrays()}

        return {
            "base_time": base_time_key,
            "lead_time": lead_time_key,
            "outputs": outputs,
            "attrs": dict(group.attrs)
        }

    def forward(self, sample: dict) -> dict:
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
                    if tensor is not None:
                        data_np = tensor[i].detach().cpu().numpy()
                        lead_group.array(name, data=data_np, overwrite=True, fill_value=None)

                print(f"Saved: {self.path}/{base_t_key}/{lead_t_key}")

            self.index_array.append(paths_to_append)
        else:
            base_t_key = str(base_times)
            lead_t_key = str(lead_times)

            lead_group = self.root.require_group(base_t_key).require_group(lead_t_key)

            for name, tensor in sample.items():
                if tensor is not None:
                    data_np = tensor.detach().cpu().numpy()
                    lead_group.array(name, data=data_np, overwrite=True, fill_value=None)

            print(f"Saved: {self.path}/{base_t_key}/{lead_t_key}")

            self.index_array.append([[base_t_key, lead_t_key]])

        return sample

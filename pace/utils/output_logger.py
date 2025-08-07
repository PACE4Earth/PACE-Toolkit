import os
from mpi4py import MPI
import shutil

import zarr
import numpy as np
import numcodecs

import torch
import torch.nn as nn

class MPIZarrSaver:
    def __init__(self, path: str, comm):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.path = path

        # Step 1: Rank 0 cleans up and initializes the archive.
        if self.rank == 0:
            print(f"Rank 0: Initializing Zarr archive and index at {self.path}")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            
            # Open the store briefly to create the index array structure.
            # No lock is needed here since only rank 0 is running this block.
            with zarr.open_group(self.path, mode='w') as root:
                root.create_dataset(
                    '_index',
                    shape=(0, 2),
                    chunks=(1024, 2),
                    dtype=object,
                    object_codec=numcodecs.JSON()
                )
        
        # Step 2: All processes must wait for rank 0 to finish.
        self.comm.Barrier()

        # Step 3: All processes open the Zarr store with a file-based lock.
        # The lock file is created inside the Zarr directory.
        lock_path = os.path.join(self.path, '.zarrlock')
        lock = zarr.ProcessSynchronizer(lock_path)
        
        self.saver = IndexedZarrSaver(self.path, synchronizer=lock)
        if self.rank == 0:
            print(f"All {self.size} ranks have opened the synchronized Zarr archive.")

    def save(self, sample: dict):
        """A wrapper for the forward call to save a sample from any rank."""
        print(f"Rank {self.rank}: Saving data for base_time {sample['base_time']}")
        self.saver(sample)

    def get_saver_instance(self):
        """Returns the underlying saver object for inspection, e.g., len()."""
        return self.saver

class IndexedZarrSaver(nn.Module):
    """
    A torch.nn.Module that saves forecasts and provides integer-based access. 

    This module saves data to a hierarchical structure: `root/base_time/lead_time/`,
    where lead_time is saved in the format '240h'.

    It also maintains an internal `_index` array within the Zarr store, mapping
    an integer index to each (base_time, lead_time) pair.

    This allows the saver instance to be used like a dataset for easy retrieval:
    `data = saver[0]`
    """
    def __init__(self, path: str, synchronizer=None):
        
        super().__init__()
        
        self.path = path  # e.g., outputs/graphcast.zarr

        self.root = zarr.open_group(self.path, mode='a', synchronizer=synchronizer)

        self.index_array = self.root['_index']

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

    def format_lead_time(self, lt):
        # Convert timedelta to hours string like "240h"
        if isinstance(lt, torch.Tensor):
            lt = lt.item()
        hours = int(lt.total_seconds() // 3600)
        return f"{hours}h"

    def forward(self, sample: dict) -> dict:
        base_times = sample.pop('base_time')
        lead_times = sample.pop('lead_time')

        is_batch = isinstance(base_times, (list, tuple))
        new_indices = []

        if is_batch:
            paths_to_append = []
            for i in range(len(base_times)):
                base_t_key = str(base_times[i])
                lead_t_key = self.format_lead_time(lead_times[i])
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
            lead_t_key = self.format_lead_time(lead_times)

            lead_group = self.root.require_group(base_t_key).require_group(lead_t_key)

            for name, tensor in sample.items():
                if tensor is not None:
                    data_np = tensor.detach().cpu().numpy()
                    lead_group.array(name, data=data_np, overwrite=True, fill_value=None)

            print(f"Saved: {self.path}/{base_t_key}/{lead_t_key}")

            self.index_array.append([[base_t_key, lead_t_key]])

        return sample

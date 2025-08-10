import os
from mpi4py import MPI
import shutil
import warnings
from json import JSONDecodeError

import zarr
import numpy as np
import numcodecs

import torch
import torch.nn as nn
from torch.utils.data import Dataset

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
            # with zarr.open_group(self.path, mode='w') as root:
            #     root.create_dataset(
            #         '_index',
            #         shape=(0, 3),
            #         chunks=(1024, 2),
            #         dtype=object,``
            #         object_codec=numcodecs.JSON()
            #     )
        
        # Step 2: All processes must wait for rank 0 to finish.
        self.comm.Barrier()

        # Step 3: All processes open the Zarr store with a file-based lock.
        # The lock file is created inside the Zarr directory.
        lock_path = os.path.join(self.path, '.zarrlock')
        lock = zarr.ProcessSynchronizer(lock_path)
        
        self.saver = ZarrHandler(self.path, synchronizer=lock)
        if self.rank == 0:
            print(f"All {self.size} ranks have opened the synchronized Zarr archive.")

    def save(self, sample: dict):
        """A wrapper for the forward call to save a sample from any rank."""
        # print(f"Rank {self.rank}: Saving data for base_time {sample['base_time']}")
        self.saver(sample)

    def get_saver_instance(self):
        """Returns the underlying saver object for inspection, e.g., len()."""
        return self.saver

class ZarrHandler(nn.Module):
    """
    A torch.nn.Module that saves forecasts and provides integer-based access. 

    This module saves data to a hierarchical structure: `root/base_time/lead_time/`,
    where lead_time is saved in the format '240h'.

    It also maintains an internal `_index` array within the Zarr store, mapping
    an integer index to each (base_time, lead_time) pair.

    This allows the saver instance to be used like a dataset for easy retrieval:
    `data = saver[0]`
    """
    def __init__(self, path: str, mode='a', synchronizer=None):
        
        super().__init__()
        
        self.path = path  # e.g., outputs/graphcast.zarr

        self.root = zarr.open_group(self.path, mode=mode, synchronizer=synchronizer)

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

        else:
            base_t_key = str(base_times)
            lead_t_key = self.format_lead_time(lead_times)

            lead_group = self.root.require_group(base_t_key).require_group(lead_t_key)

            for name, tensor in sample.items():
                if tensor is not None:
                    data_np = tensor.detach().cpu().numpy()
                    lead_group.array(name, data=data_np, overwrite=True, fill_value=None)

            print(f"Saved: {self.path}/{base_t_key}/{lead_t_key}")

        return sample

class ZarrDataset(Dataset):
    """
    A PyTorch Dataset for accessing a Zarr store with a specific hierarchical
    structure: `root/{base_time}/{lead_time}/{variable}`.

    Each item in the dataset corresponds to a unique (base_time, lead_time)
    combination.
    """
    def __init__(self, path, variables=None):
        """
        Args:
            zarr_path (str): Path to the root Zarr directory (e.g., 'era5.zarr').
            variables (list of str, optional): A specific list of variables to load.
                                               If None, all variables found in the
                                               first sample are loaded.
        """
        self.zarr_path = path
        self.root = zarr.open_group(self.zarr_path, mode='r')
        self.samples = self._create_sample_map()
        
        if not self.samples:
            raise ValueError("No samples found in the Zarr store. Check the path and structure.")

        if variables:
            self.variables = variables
        else:
            # Auto-discover variables from the first sample if not provided
            first_base, first_lead = self.samples[0]
            self.variables = list(self.root[first_base][first_lead].keys())
            print(self.variables)


    def _create_sample_map(self):
        """
        Scans the Zarr store to find all (base_time, lead_time) pairs,
        ignoring hidden directories like .zarrlock.
        """
        samples = []
        for base_time in self.root.keys():
            # --> ADD THIS CHECK <--
            if base_time.startswith('.'):
                continue  # Skip hidden files/directories like .zarrlock

            try:
                base_time_group = self.root[base_time]
                for lead_time in base_time_group.keys():
                    samples.append((base_time, lead_time))
            except (JSONDecodeError, KeyError) as e:
                warnings.warn(
                    f"Skipping corrupted or invalid entry in Zarr store: '{base_time}'. "
                    f"Error: {e}"
                )
                continue
        return samples

    def __len__(self):
        """Returns the total number of (base_time, lead_time) samples."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            dict: A dictionary containing the 'base_time', 'lead_time',
                  and a 'data' dictionary where keys are variable names
                  and values are the corresponding PyTorch tensors.
        """
        if not 0 <= idx < len(self.samples):
            raise IndexError("Index out of range")

        base_time, lead_time = self.samples[idx]
        
        output = {}
        
        output['base_time'] = base_time
        output['lead_time'] = lead_time
        
        data_group = self.root[base_time][lead_time]

        data_tensors = {}
        for var_name in self.variables:

            zarr_array = data_group[var_name]
            
            # print(var_name, type(zarr_array), zarr_array.shape)
            
            numpy_array = zarr_array[:]
            output[var_name] = torch.from_numpy(numpy_array)
            
        return output


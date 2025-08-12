import os
from mpi4py import MPI
import shutil
import warnings
from json import JSONDecodeError
from datetime import timedelta

import zarr
import xarray as xr
import numpy as np
import numcodecs

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MPIZarrSaver:
    def __init__(self, path: str, comm, lat=None, lon=None):
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.path = path
        self.lat = lat
        self.lon = lon

        # Step 1: Rank 0 cleans up and initializes the archive.
        if self.rank == 0:
            print(f"Rank 0: Initializing Zarr archive and index at {self.path}")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)

            self.root = zarr.open_group(self.path, mode='w')

            # zarr.open_group(self.path, mode='w')
            
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
        
        # self.saver = ZarrHandler(self.path, synchronizer=lock, lat=lat, lon=lon)
        self.saver = XarrayZarrHandler(self.path, synchronizer=lock, lat=lat, lon=lon)
        if self.rank == 0:
            print(f"All {self.size} ranks have opened the synchronized Zarr archive.")

    def save(self, sample: dict):
        """A wrapper for the forward call to save a sample from any rank."""
        # print(f"Rank {self.rank}: Saving data for base_time {sample['base_time']}")
        self.saver(sample)

    def get_saver_instance(self):
        """Returns the underlying saver object for inspection, e.g., len()."""
        return self.saver

    
    def initialize_store(self, sample: dict):
        """Initializes the Zarr arrays and coordinates based on the first data sample."""
        # Create static coordinate arrays for latitude and longitude
        # self.root.create_dataset(
        #     'lat', 
        #     data=np.array(self.lat), 
        #     object_codec=numcodecs.Pickle(), 
        #     overwrite=True,
        # )
        # self.root.create_dataset(
        #     'lon', 
        #     data=np.array(self.lon),
        #     object_codec=numcodecs.Pickle(),  
        #     overwrite=True,
        # )
        
        # # Create appendable coordinate arrays for the index dimension
        # self.root.create_dataset('base_time', shape=(0,), chunks=(1024,), dtype='datetime64[ns]', overwrite=True)
        # self.root.create_dataset('lead_time', shape=(0,), chunks=(1024,), dtype='timedelta64[h]', overwrite=True)

        arr = self.root.create_dataset(
            'lat', 
            data=np.array(self.lat),
            object_codec=numcodecs.Pickle(),
            overwrite=True,
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['lat']

        arr = self.root.create_dataset(
            'lon', 
            data=np.array(self.lon),
            object_codec=numcodecs.Pickle(),
            overwrite=True,
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['lon']

        # index coordinates
        arr = self.root.create_dataset('base_time', shape=(0,), chunks=(1024,), dtype='datetime64[ns]', overwrite=True)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['base_time']

        arr = self.root.create_dataset('lead_time', shape=(0,), chunks=(1024,), dtype='timedelta64[h]', overwrite=True)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['lead_time']

        # Create an appendable data array for each variable
        for name, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                
                # tensor = tensor.squeeze(0)
                spatial_dims = len(tensor.shape)
                
                if spatial_dims == 2: # (lat, lon)
                    dims = ['lat', 'lon']
                elif spatial_dims == 3: # (level, lat, lon)
                    dims = ['level', 'lat', 'lon']
                else: # Fallback for other dimensions
                    dims = [f'dim_{i}' for i in range(spatial_dims)]
                    
                shape = tensor.shape  # Start with size 0 on the 'idx' dimension
                chunks = tensor.shape # Chunk by each item
                arr = self.root.create_dataset(
                    name, 
                    shape=shape, 
                    chunks=chunks, 
                    dtype=np.array(tensor).dtype, 
                    overwrite=True,
                )
                # This metadata is essential for xarray
                arr.attrs['_ARRAY_DIMENSIONS'] = dims
                # print(name, '\t', tensor.shape, '\t', dims)
            # else:
            #     print('\nNon-Tensor entries:')
            #     print(name, type(tensor))
        
        self._initialized = True
        print(f"Zarr store initialized for xarray at: {self.path}")
        try:
            zarr.consolidate_metadata(self.path)
            # print('Did we win?')
        except Exception as e:
            print(e)
        

class XarrayZarrHandler(nn.Module):
    """
    A torch.nn.Module that saves forecasts to a Zarr store in an xarray-compatible format.

    This module saves data variables as appendable arrays along a shared 'idx' dimension.
    It creates a flat Zarr store with datasets for each variable and corresponding 
    coordinates, which can be opened directly with `xarray.open_zarr()`.
    
    It is designed as a plug-and-play replacement for handlers that save data
    in a nested format, with the added requirement of latitude and longitude coordinates
    at initialization.
    """
    def __init__(self, path: str, lat: np.ndarray, lon: np.ndarray, mode='a', synchronizer=None):
        """
        Initializes the handler.

        Args:
            path (str): The path to the output Zarr store.
            lat (np.ndarray): A 1D array of latitude values.
            lon (np.ndarray): A 1D array of longitude values.
            mode (str, optional): The mode for opening the Zarr store. Defaults to 'a'.
            synchronizer (zarr.sync.Synchronizer, optional): For concurrent writing.
        """
        super().__init__()
        self.path = path
        self.lat = lat
        self.lon = lon
        self.root = zarr.open_group(self.path, mode=mode, synchronizer=synchronizer)
        # self._initialized = False

    def _to_timedelta(self, lt: timedelta) -> np.timedelta64:
        """Converts a datetime.timedelta to a numpy.timedelta64 in hours."""
        if isinstance(lt, torch.Tensor):
            lt = lt.item()
        hours = int(lt.total_seconds() // 3600)
        return np.timedelta64(hours, 'h')

    def forward(self, sample: dict) -> dict:
        
        # print(sample.keys())
        
        # """Saves a batch of forecasts to the Zarr store."""
        # # On the first call, set up the entire Zarr store structure
        # if not self._initialized:
        #     tensor_sample = {k: v for k, v in sample.items() if isinstance(v, torch.Tensor)}
        #     if not tensor_sample:
        #         raise ValueError("Sample contains no tensors to initialize the Zarr store.")
        #     self._initialize_store(tensor_sample)

        # for name, data in sample.items():
            
        #     if isinstance(data, torch.Tensor):
        #         data = data.squeeze(0)
            
        #     data_np = np.array(data)
            
        #     # Add this debugging block!
        #     try: 
        #         if self.root[name].shape[1:] != data_np.shape[1:]:
        #             print(f"!! SHAPE MISMATCH DETECTED FOR '{name}' !!")
        #             print(f"  Expected shape (from init): {self.root[name].shape}")
        #             print(f"  Received shape to append:   {data_np.shape}") # Show what append expects
        #             print(f"  Actual data shape:          {data_np.shape}")
        #     except Exception as e:
        #         print(e)

        base_times = sample.pop('base_time')
        lead_times = sample.pop('lead_time')
        
        # Standardize input to be a batch
        is_single_item = not isinstance(base_times, (list, tuple))
        if is_single_item:
            base_times = [base_times]
            lead_times = [lead_times]

        # Append time coordinates
        self.root['base_time'].append(np.array(base_times, dtype='datetime64[ns]'))
        self.root['lead_time'].append(np.array([self._to_timedelta(lt) for lt in lead_times]))

        # # Append variable data
        # for name, tensor in sample.items():
        #     if tensor != None:
        #         data_np = tensor.detach().cpu().numpy()
        #         # Add a batch dimension if the input was a single item
        #         if is_single_item:
        #             data_np = data_np[np.newaxis, ...]
        #         self.root[name].append(data_np)
        
        for name, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                try:
                    
                    # tensor = tensor.squeeze(0)
                    data_np = tensor.detach().cpu().numpy()
                    
                    # print(name, data_np.shape)
                    
                    self.root[name].append(data_np)
                    
                except ValueError:
                    # If a shape mismatch occurs, catch it, log it, and continue
                    print(f"⚠️  WARNING: Shape mismatch for variable '{name}'. Skipping this data point.")
                    print(f"   - Expected shape (from existing array): {self.root[name].shape}")
                    print(f"   - Received shape (from new data):     {data_np.shape}")
                    print("   - The program will continue running.")
                    continue # Explicitly move to the next item in the loop
            
        # print(f"Saved {len(base_times)} item(s) to {self.path}")
        return sample

class ZarrDataset(Dataset):
    """
    A PyTorch Dataset for accessing a Zarr store with a specific hierarchical
    structure: `root/{base_time}/{lead_time}/{variable}`.

    Each item in the dataset corresponds to a unique (base_time, lead_time)
    combination.
    """
    def __init__(self, path, variables=None, lat=None, lon=None):
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


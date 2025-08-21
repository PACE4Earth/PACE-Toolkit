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
    def __init__(self, path: str, comm, lat=None, lon=None, level=None):
        
        self._initialized = False
        
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.path = path
        self.lat = lat
        self.lon = lon
        self.level = level

        if self.rank == 0:
            print(f"Rank 0: Initializing Zarr archive and index at {self.path}")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)

            self.root = zarr.open_group(self.path, mode='w')
        
        self.comm.Barrier()

        lock_path = os.path.join(self.path, '.zarrlock')
        lock = zarr.ProcessSynchronizer(lock_path)
        
        self.saver = XarrayZarrHandler(self.path, synchronizer=lock, lat=lat, lon=lon, level=level, rank=self.rank)
        if self.rank == 0:
            print(f"All {self.size} ranks have opened the synchronized Zarr archive.")

    def save(self, sample: dict):
        self.saver(sample)

    def get_saver_instance(self):
        return self.saver

    def initialize_store(self, sample: dict):
        if self._initialized:
            print(f"{self.rank}: Attempted re-initialization for xarray at: {self.path}")
            return

        arr = self.root.create_dataset(
            'lat', data=np.array(self.lat), dtype='f4', overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['lat']

        arr = self.root.create_dataset(
            'lon', data=np.array(self.lon), dtype='f4', overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['lon']

        if self.level is not None:
            arr = self.root.create_dataset(
                'level', data=np.array(self.level), dtype='i4', overwrite=True
            )
            arr.attrs['_ARRAY_DIMENSIONS'] = ['level']

        # idx coordinate
        arr = self.root.create_dataset(
            'idx', shape=(0,), chunks=(1024,), dtype='i8', overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']

        # base_time coordinate (align with idx)
        arr = self.root.create_dataset(
            'base_time', shape=(0,), chunks=(1024,), dtype='datetime64[ns]', overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']
        arr.attrs['coordinates'] = 'base_time'

        # lead_time coordinate (align with idx)
        arr = self.root.create_dataset(
            'lead_time', shape=(0,), chunks=(1024,), dtype='timedelta64[h]', overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']
        arr.attrs['coordinates'] = 'lead_time'

        for name, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.ndim == 4:
                    tensor = tensor[0]
                    if (tensor.shape[0] == 1) and (self.level != None):
                        tensor = tensor.expand(np.array(self.level).shape[0], -1, -1)
                        
                elif tensor.ndim == 3:
                    tensor = tensor[0]
                else:
                    continue

                shape = (0,) + tensor.shape
                chunks = (1,) + tensor.shape
                arr = self.root.create_dataset(
                    name,
                    shape=shape,
                    chunks=chunks,
                    dtype=np.array(tensor.cpu()).dtype,
                    overwrite=True,
                )
                
                if tensor.ndim == 3:
                    arr.attrs['_ARRAY_DIMENSIONS'] = ['idx', 'level', 'lat', 'lon']
                    arr.attrs['coordinates'] = 'idx base_time lead_time lat lon level'
                elif tensor.ndim == 2:
                    arr.attrs['_ARRAY_DIMENSIONS'] = ['idx', 'var_1', 'var_2']
                    arr.attrs['coordinates'] = 'idx base_time lead_time var_1 var_2'
        
        self._initialized = True
        print(f"{self.rank}: Zarr store initialized for xarray at: {self.path}")
        try:
            zarr.consolidate_metadata(self.path)
        except Exception as e:
            print(e)
        

class XarrayZarrHandler(nn.Module):
    def __init__(self, path: str, lat: np.ndarray, lon: np.ndarray, level=None, mode='a', synchronizer=None, rank=-1):
        super().__init__()
        self.path = path
        self.rank = rank
        self.lat = lat
        self.lon = lon
        self.level = level
        self.root = zarr.open_group(self.path, mode=mode, synchronizer=synchronizer)

    def _to_timedelta(self, lt: timedelta) -> np.timedelta64:
        if isinstance(lt, torch.Tensor):
            lt = lt.item()
        hours = int(lt.total_seconds() // 3600)
        return np.timedelta64(hours, 'h')

    def forward(self, sample: dict) -> dict:
        base_times = sample.pop('base_time')
        lead_times = sample.pop('lead_time')
        indices = sample.pop('idx', None)

        is_single_item = not isinstance(base_times, (list, tuple))
        if is_single_item:
            base_times = [base_times]
            lead_times = [lead_times]
            if indices is not None:
                indices = [indices]

        # First, append the metric data
        for name, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                try:
                    if tensor.ndim == 4:
                        tensor = tensor[0]
                        if (tensor.shape[0] == 1) and (self.level != None):
                            tensor = tensor.expand(np.array(self.level).shape[0], -1, -1)
                    elif tensor.ndim == 3:
                        # print(name, tensor.shape)
                        tensor = tensor[0]
                    data_np = tensor.detach().cpu().numpy()
                    data_np = np.expand_dims(data_np, axis=0)  # add idx dimension
                    self.root[name].append(data_np)
                except ValueError as e:
                    print(f"Rank {self.rank}: FAILED to append {name}. Shape mismatch. Error: {e}")
                    return sample  # skip saving coordinates if metric fails

        # Then, append coordinates just once per sample
        self.root['base_time'].append(np.array(base_times, dtype='datetime64[ns]'))
        self.root['lead_time'].append(np.array([self._to_timedelta(lt) for lt in lead_times]))
        
        if indices is not None:
            indices_np = np.array([idx.item() if isinstance(idx, torch.Tensor) else idx for idx in indices], dtype='i8')
            self.root['idx'].append(indices_np)
        else:
            indices_np = np.array([-1], dtype='i8')
            self.root['idx'].append(indices_np)

        idx_val = indices_np[0] if indices_np.size > 0 else 'N/A'
        print(f"Rank {self.rank}: Saved idx {idx_val} for {base_times[0]} to {self.path}")

        return sample


# import os
# from mpi4py import MPI
# import shutil
# import warnings
# from json import JSONDecodeError
# from datetime import timedelta

# import zarr
# import xarray as xr
# import numpy as np
# import numcodecs

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset

# class MPIZarrSaver:
#     def __init__(self, path: str, comm, lat=None, lon=None):
        
#         self._initialized = False
        
#         self.comm = comm
#         self.rank = self.comm.Get_rank()
#         self.size = self.comm.Get_size()
#         self.path = path
#         self.lat = lat
#         self.lon = lon

#         # Step 1: Rank 0 cleans up and initializes the archive.
#         if self.rank == 0:
#             print(f"Rank 0: Initializing Zarr archive and index at {self.path}")
#             if os.path.exists(self.path):
#                 shutil.rmtree(self.path)

#             self.root = zarr.open_group(self.path, mode='w')

#             # zarr.open_group(self.path, mode='w')
            
#             # Open the store briefly to create the index array structure.
#             # No lock is needed here since only rank 0 is running this block.
#             # with zarr.open_group(self.path, mode='w') as root:
#             #     root.create_dataset(
#             #         '_index',
#             #         shape=(0, 3),
#             #         chunks=(1024, 2),
#             #         dtype=object,``
#             #         object_codec=numcodecs.JSON()
#             #     )
        
#         # Step 2: All processes must wait for rank 0 to finish.
#         self.comm.Barrier()

#         # Step 3: All processes open the Zarr store with a file-based lock.
#         # The lock file is created inside the Zarr directory.
#         lock_path = os.path.join(self.path, '.zarrlock')
#         lock = zarr.ProcessSynchronizer(lock_path)
        
#         # self.saver = ZarrHandler(self.path, synchronizer=lock, lat=lat, lon=lon)
#         self.saver = XarrayZarrHandler(self.path, synchronizer=lock, lat=lat, lon=lon, rank=self.rank)
#         if self.rank == 0:
#             print(f"All {self.size} ranks have opened the synchronized Zarr archive.")

#     def save(self, sample: dict):
#         """A wrapper for the forward call to save a sample from any rank."""
#         # print(f"Rank {self.rank}: Saving data for base_time {sample['base_time']}")
#         self.saver(sample)

#     def get_saver_instance(self):
#         """Returns the underlying saver object for inspection, e.g., len()."""
#         return self.saver

    
#     def initialize_store(self, sample: dict):
#         """Initializes the Zarr arrays and coordinates based on the first data sample."""
        
#         if self._initialized:
#             print(f"{self.rank}: Attempted re-initialization for xarray at: {self.path}")
#             return

#         arr = self.root.create_dataset(
#             'lat', 
#             data=np.array(self.lat),
#             object_codec=numcodecs.Pickle(),
#             overwrite=True,
#         )
#         arr.attrs['_ARRAY_DIMENSIONS'] = ['lat']

#         arr = self.root.create_dataset(
#             'lon', 
#             data=np.array(self.lon),
#             object_codec=numcodecs.Pickle(),
#             overwrite=True,
#         )
#         arr.attrs['_ARRAY_DIMENSIONS'] = ['lon']


#         # index coordinates
#         arr = self.root.create_dataset('idx', shape=(0,), chunks=(1024,), dtype='float', overwrite=True)
#         arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']

#         # index coordinates
#         arr = self.root.create_dataset('base_time', shape=(0,), chunks=(1024,), dtype='datetime64[ns]', overwrite=True)
#         arr.attrs['_ARRAY_DIMENSIONS'] = ['base_time']

#         arr = self.root.create_dataset('lead_time', shape=(0,), chunks=(1024,), dtype='timedelta64[h]', overwrite=True)
#         arr.attrs['_ARRAY_DIMENSIONS'] = ['lead_time']

#         # Create an appendable data array for each variable
#         for name, tensor in sample.items():
#             if isinstance(tensor, torch.Tensor):
                
#                 # tensor = tensor.squeeze(0)
#                 spatial_dims = len(tensor.shape)
                
#                 if spatial_dims == 2: # (lat, lon)
#                     dims = ['lat', 'lon']
#                 elif spatial_dims == 3: # (level, lat, lon)
#                     dims = ['level', 'lat', 'lon']
#                 else: # Fallback for other dimensions
#                     dims = [f'dim_{i}' for i in range(spatial_dims)]
                    
#                 shape = tensor.shape  # Start with size 0 on the 'idx' dimension
#                 chunks = tensor.shape # Chunk by each item
#                 arr = self.root.create_dataset(
#                     name, 
#                     shape=shape, 
#                     chunks=chunks, 
#                     dtype=np.array(tensor.cpu()).dtype, 
#                     overwrite=True,
#                 )
#                 # This metadata is essential for xarray
#                 arr.attrs['_ARRAY_DIMENSIONS'] = dims
#                 # print(name, '\t', tensor.shape, '\t', dims)
#             # else:
#             #     print('\nNon-Tensor entries:')
#             #     print(name, type(tensor))
        
#         self._initialized = True
#         print(f"{self.rank}: Zarr store initialized for xarray at: {self.path}")
#         try:
#             zarr.consolidate_metadata(self.path)
#             # print('Did we win?')
#         except Exception as e:
#             print(e)
        

# class XarrayZarrHandler(nn.Module):
#     """
#     A torch.nn.Module that saves forecasts to a Zarr store in an xarray-compatible format.

#     This module saves data variables as appendable arrays along a shared 'idx' dimension.
#     It creates a flat Zarr store with datasets for each variable and corresponding 
#     coordinates, which can be opened directly with `xarray.open_zarr()`.
    
#     It is designed as a plug-and-play replacement for handlers that save data
#     in a nested format, with the added requirement of latitude and longitude coordinates
#     at initialization.
#     """
#     def __init__(self, path: str, lat: np.ndarray, lon: np.ndarray, mode='a', synchronizer=None, rank=-1):
#         """
#         Initializes the handler.

#         Args:
#             path (str): The path to the output Zarr store.
#             lat (np.ndarray): A 1D array of latitude values.
#             lon (np.ndarray): A 1D array of longitude values.
#             mode (str, optional): The mode for opening the Zarr store. Defaults to 'a'.
#             synchronizer (zarr.sync.Synchronizer, optional): For concurrent writing.
#         """
#         super().__init__()
#         self.path = path
#         self.rank = rank
#         self.lat = lat
#         self.lon = lon
#         self.root = zarr.open_group(self.path, mode=mode, synchronizer=synchronizer)
#         # self._initialized = False

#     def _to_timedelta(self, lt: timedelta) -> np.timedelta64:
#         """Converts a datetime.timedelta to a numpy.timedelta64 in hours."""
#         if isinstance(lt, torch.Tensor):
#             lt = lt.item()
#         hours = int(lt.total_seconds() // 3600)
#         return np.timedelta64(hours, 'h')

#     def forward(self, sample: dict) -> dict:
#         # --- Pop ALL coordinates before the loop ---
#         base_times = sample.pop('base_time')
#         lead_times = sample.pop('lead_time')
#         indices = sample.pop('idx', None) # Safely pop idx

#         is_single_item = not isinstance(base_times, (list, tuple))
#         if is_single_item:
#             base_times = [base_times]
#             lead_times = [lead_times]
#             if indices is not None:
#                 indices = [indices]

#         all_tensors_saved_successfully = True
        
#         # This loop now only processes actual data variables
#         for name, tensor in sample.items():
#             if isinstance(tensor, torch.Tensor):
#                 try:
#                     data_np = tensor.detach().cpu().numpy()
#                     self.root[name].append(data_np)
#                 except ValueError as e:
#                     print(f"Rank {self.rank}: FAILED to append {name}. Shape mismatch. Error: {e}")
#                     all_tensors_saved_successfully = False
#                     break 
        
#         # --- Atomically append ALL coordinates if the data was saved ---
#         if all_tensors_saved_successfully:
#             self.root['base_time'].append(np.array(base_times, dtype='datetime64[ns]'))
#             self.root['lead_time'].append(np.array([self._to_timedelta(lt) for lt in lead_times]))
#             if indices is not None:
#                 # Ensure indices is a numpy array before appending
#                 indices_np = np.array([idx.item() if isinstance(idx, torch.Tensor) else idx for idx in indices])
#                 self.root['idx'].append(indices_np)
            
#             idx_val = indices_np[0] if indices is not None else 'N/A'
#             print(f"Rank {self.rank}: Saved idx {idx_val} for {base_times[0]} to {self.path}")
#         else:
#             print(f"XXX Rank {self.rank}: Discarded coordinates for {base_times[0]} due to data append failure.")
        
#         return sample

    # def forward(self, sample: dict) -> dict:
        
    #     base_times = sample.pop('base_time')
    #     lead_times = sample.pop('lead_time')
        
    #     is_single_item = not isinstance(base_times, (list, tuple))
    #     if is_single_item:
    #         base_times = [base_times]
    #         lead_times = [lead_times]
        
    #     # Keep a reference to the idx for logging, if it exists
    #     idx_for_logging = sample.get('idx', -1)
        
    #     is_single_item = not isinstance(base_times, (list, tuple))
    #     if is_single_item:
    #         base_times = [base_times]
    #         lead_times = [lead_times]
        
    #     # This flag will track if ALL tensors in the sample are saved
    #     all_tensors_saved_successfully = True
        
    #     for name, tensor in sample.items():
    #         if isinstance(tensor, torch.Tensor):
    #             try:
    #                 data_np = tensor.detach().cpu().numpy()
    #                 self.root[name].append(data_np)
    #             except ValueError as e:
    #                 # If any append fails, log it and mark the entire sample as failed
    #                 print(f"Rank {self.rank}: FAILED to append {name}. Expected shape ~{self.root[name].shape[1:]} but got {data_np.shape}. Error: {e}")
    #                 all_tensors_saved_successfully = False
    #                 # No need to process other tensors for this sample, break out
    #                 break 
        
    #     # ONLY append coordinates if the entire sample was saved without errors
    #     if all_tensors_saved_successfully:
    #         self.root['base_time'].append(np.array(base_times, dtype='datetime64[ns]'))
    #         self.root['lead_time'].append(np.array([self._to_timedelta(lt) for lt in lead_times]))
    #         if isinstance(idx_for_logging, torch.Tensor):
    #             idx_for_logging = idx_for_logging.detach().cpu().numpy()
    #         print(f"Rank {self.rank}: Saved {idx_for_logging} for {base_times} to {self.path}")
    #     else:
    #         print(f"XXX Rank {self.rank}: Discarded coordinates for {base_times} due to data append failure in {self.path}")

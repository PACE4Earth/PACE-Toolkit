import os
from mpi4py import MPI
import shutil
from datetime import timedelta

import zarr
import xarray as xr
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MPIZarrSaver:
    def __init__(self, path: str, comm, lat=None, lon=None, level=None, N_total=None):  
        self._initialized = False
        
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.path = path
        self.lat = lat
        self.lon = lon
        self.level = level
        self.N_total = N_total  

        if self.rank == 0:
            print(f"Rank 0: Initializing Zarr archive and index at {self.path}")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)

            self.root = zarr.open_group(self.path, mode='w')
        
        self.comm.Barrier()

        lock_path = os.path.join(self.path, '.zarrlock')
        lock = zarr.ProcessSynchronizer(lock_path)
        
        self.saver = XarrayZarrHandler(
            self.path,
            synchronizer=lock,
            lat=lat,
            lon=lon,
            level=level,
            rank=self.rank,
            comm=self.comm,       
            N_total=self.N_total  
        )
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

        # Preallocate idx coordinate
        arr = self.root.create_dataset(
            'idx', shape=(self.N_total,), chunks=(1024,), dtype='i8', fill_value=-1, overwrite=True
        )
        arr[:] = np.arange(self.N_total, dtype='i8')  # not nice but works
        arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']

        # base_time coordinate (as coordinate of idx)
        arr = self.root.create_dataset(
            'base_time', shape=(self.N_total,), chunks=(1024,), dtype='datetime64[ns]', overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']

        # lead_time coordinate (as coordinate of idx)
        arr = self.root.create_dataset(
            'lead_time', shape=(self.N_total,), chunks=(1024,), dtype='timedelta64[h]', overwrite=True
        )
        arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']

        # Preallocate metric arrays
        for name, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.ndim == 4:
                    tensor = tensor[0]
                    if self.level is not None:
                        if (tensor.shape[0] == 1) and (np.array(self.level).shape[0] != 1):
                            tensor = tensor.expand(np.array(self.level).shape[0], -1, -1)
                        
                elif tensor.ndim == 3:
                    tensor = tensor[0]
                else:
                    continue

                shape = (self.N_total,) + tensor.shape   # preallocate with N_total
                chunks = (1,) + tensor.shape

                arr = self.root.create_dataset(
                    name, shape=shape, chunks=chunks, dtype=np.array(tensor.cpu()).dtype, fill_value=None, overwrite=True
                )

                if tensor.ndim == 3:
                    arr.attrs['_ARRAY_DIMENSIONS'] = ['idx', 'level', 'lat', 'lon']
                    arr.attrs['coordinates'] = 'base_time lead_time'
                elif tensor.ndim == 2:
                    arr.attrs['_ARRAY_DIMENSIONS'] = ['idx', 'var_1', 'var_2']
                    arr.attrs['coordinates'] = 'base_time lead_time'

        self._initialized = True
        print(f"{self.rank}: Zarr store initialized for xarray at: {self.path}")
        zarr.consolidate_metadata(self.path)

class XarrayZarrHandler(nn.Module):
    def __init__(self, path: str, lat: np.ndarray, lon: np.ndarray, level=None,
                 mode='a', synchronizer=None, rank=-1, comm=None, N_total=None):  
        super().__init__()
        self.path = path
        self.rank = rank
        self.lat = lat
        self.lon = lon
        self.level = level
        self.comm = comm
        self.N_total = N_total
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

        # Compute actual indices to write into
        if indices is not None:
            indices_np = np.array([idx.item() if isinstance(idx, torch.Tensor) else idx for idx in indices], dtype='i8')
        else:
            # fallback if idx not provided
            indices_np = np.arange(len(base_times), dtype='i8')

        # Write metric data directly at global positions
        for name, tensor in sample.items():
            if isinstance(tensor, torch.Tensor):
                if tensor.ndim == 4:
                    tensor = tensor[0]
                    if (tensor.shape[0] == 1) and (self.level is not None):
                        tensor = tensor.expand(np.array(self.level).shape[0], -1, -1)
                elif tensor.ndim == 3:
                    tensor = tensor[0]
                data_np = tensor.detach().cpu().numpy()
                data_np = np.expand_dims(data_np, axis=0)  # (1,...)
                self.root[name][indices_np] = data_np  # write at actual idx


        # Write coordinates
        self.root['base_time'][indices_np] = np.array(base_times, dtype='datetime64[ns]')
        self.root['lead_time'][indices_np] = np.array([self._to_timedelta(lt) for lt in lead_times])
        self.root['idx'][indices_np] = indices_np

        return sample

import os
import shutil
from datetime import timedelta

import zarr
import numpy as np
import torch
import torch.nn as nn

class MPIZarrSaver:
    """
    MPI-aware wrapper around Zarr saving for distributed metric evaluation.

    This class is responsible for:
    - Initializing a shared Zarr store (rank 0 creates/removes existing store).
    - Setting up synchronizers for safe concurrent writes.
    - Preallocating coordinates and metric arrays.
    - Delegating actual writes to `XarrayZarrHandler`.

    Parameters
    ----------
    path : str
        Filesystem path to the Zarr store.
    comm : mpi4py.MPI.Comm
        MPI communicator for rank coordination.
    lat, lon : array-like, optional
        Latitude and longitude coordinates.
    level : array-like, optional
        Vertical levels, if applicable.
    N_total : int, optional
        Total number of samples across all workers (used for preallocation).
    """

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
            print(f"[Rank 0] Initializing Zarr archive at {self.path}\n")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            self.root = zarr.open_group(self.path, mode='w')
        
        # Ensure store exists before others proceed
        self.comm.Barrier()

        # Zarr synchronizer ensures multiple processes can write safely
        lock_path = os.path.join(self.path, '.zarrlock')
        lock = zarr.ProcessSynchronizer(lock_path)
        
        self.saver = XarrayZarrHandler(
            path=path,
            synchronizer=lock,
            lat=lat,
            lon=lon,
            level=level,
            rank=self.rank,
            comm=comm,       
            N_total=N_total  
        )

    def save(self, sample: dict):
        """Write a single sample into the Zarr store."""
        self.saver(sample)

    def get_saver_instance(self):
        """Return the underlying `XarrayZarrHandler` for advanced control."""
        return self.saver

    def initialize_store(self, sample: dict):
        """
        Create coordinate and variable arrays in the Zarr store.

        Called once by rank 0 before writing any data. Preallocates
        `lat`, `lon`, `level`, `idx`, `base_time`, `lead_time`, and all
        metric arrays based on the first provided `sample`.
        """

        # Coordinate datasets
        arr = self.root.create_dataset('lat', data=np.array(self.lat), dtype='f4', overwrite=True)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['lat']

        arr = self.root.create_dataset('lon', data=np.array(self.lon), dtype='f4', overwrite=True)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['lon']

        if self.level is not None:
            arr = self.root.create_dataset('level', data=np.array(self.level), dtype='i4', overwrite=True)
            arr.attrs['_ARRAY_DIMENSIONS'] = ['level']

        # Preallocate idx coordinate
        arr = self.root.create_dataset('idx', shape=(self.N_total,), chunks=(1024,), dtype='i8', fill_value=-1, overwrite=True)
        arr[:] = np.arange(self.N_total, dtype='i8')  
        arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']

        # base_time coordinate (as coordinate of idx)
        arr = self.root.create_dataset('base_time', shape=(self.N_total,), chunks=(1024,), dtype='datetime64[ns]', overwrite=True)
        arr.attrs['_ARRAY_DIMENSIONS'] = ['idx']

        # lead_time coordinate (as coordinate of idx)
        arr = self.root.create_dataset('lead_time', shape=(self.N_total,), chunks=(1024,), dtype='timedelta64[h]', overwrite=True)
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
                    
                elif tensor.ndim == 2:
                    tensor = tensor[0]
                    
                else:
                    continue

                shape = (self.N_total,) + tensor.shape   # preallocate with N_total
                chunks = (1,) + tensor.shape

                arr = self.root.create_dataset(name, shape=shape, chunks=chunks, dtype=np.array(tensor.cpu()).dtype, fill_value=None, overwrite=True)
                
                # Annotate dimension order
                if tensor.ndim == 3:
                    arr.attrs['_ARRAY_DIMENSIONS'] = ['idx', 'level', 'lat', 'lon']
                elif tensor.ndim == 2:
                    arr.attrs['_ARRAY_DIMENSIONS'] = ['idx', 'var_1', 'var_2']
                elif tensor.ndim == 1:
                    arr.attrs['_ARRAY_DIMENSIONS'] = ['idx', 'null']
                
                arr.attrs['coordinates'] = 'base_time lead_time'
                    
            elif isinstance(tensor, list) and name=='var_names':
                var1_coord = self.root.create_dataset(
                    'var_1', 
                    data=tensor, 
                    shape=(len(tensor),), 
                    dtype=str, 
                    overwrite=True
                )
                var1_coord.attrs['_ARRAY_DIMENSIONS'] = ['var_1']

                var2_coord = self.root.create_dataset(
                    'var_2', 
                    data=tensor, 
                    shape=(len(tensor),), 
                    dtype=str, 
                    overwrite=True
                )
                var2_coord.attrs['_ARRAY_DIMENSIONS'] = ['var_2']
                
        self._initialized = True
        print(f"[Rank {self.rank}] Zarr store initialized for xarray at: {self.path}\n")
        zarr.consolidate_metadata(self.path)


class XarrayZarrHandler(nn.Module):
    """
    Handles writing metric samples into an existing Zarr store.

    Each forward call writes metrics and associated coordinates
    (`base_time`, `lead_time`, `idx`) at preallocated positions.

    Parameters
    ----------
    path : str
        Zarr store path.
    lat, lon : array-like
        Grid coordinates.
    level : array-like, optional
        Vertical levels.
    mode : str, default 'a'
        Zarr file mode.
    synchronizer : zarr.Synchronizer, optional
        Used for safe parallel writes.
    rank : int
        MPI rank of this handler.
    comm : mpi4py.MPI.Comm, optional
        MPI communicator.
    N_total : int, optional
        Total number of samples across all workers.
    """

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
        """Convert Python timedelta or torch scalar to numpy timedelta64[h]."""
        if isinstance(lt, torch.Tensor):
            lt = lt.item()
        hours = int(lt.total_seconds() // 3600)
        return np.timedelta64(hours, 'h')

    def forward(self, sample: dict) -> dict:
        """
        Write a sample dictionary into the Zarr store.

        Expected keys in `sample`:
        - 'base_time': datetime or list of datetimes
        - 'lead_time': timedelta or list of timedeltas
        - 'idx': optional integer index or list of indices
        - additional metric tensors to write
        """
        base_times = sample.pop('base_time')
        lead_times = sample.pop('lead_time')
        indices = sample.pop('idx', None)

        # Normalize to lists
        if not isinstance(base_times, (list, tuple)):
            base_times, lead_times, indices = [base_times], [lead_times], [indices] if indices is not None else (None)

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
                elif tensor.ndim == 2:
                    tensor = tensor[0]
                    
                data_np = tensor.detach().cpu().numpy()
                data_np = np.expand_dims(data_np, axis=0)  # (1,...)
                self.root[name][indices_np] = data_np  # write at actual idx


        # Write coordinates
        self.root['base_time'][indices_np] = np.array(base_times, dtype='datetime64[ns]')
        self.root['lead_time'][indices_np] = np.array([self._to_timedelta(lt) for lt in lead_times])
        self.root['idx'][indices_np] = indices_np

        return sample

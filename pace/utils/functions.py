import os
import json
from mpi4py import MPI
from collections import defaultdict
import time

import numpy as np
import xarray as xr
import zarr

import xarray.backends.zarr
from xarray.core.utils import is_dict_like

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from utils.dataset import UnifiedDataset
from utils.output_logger import MPIZarrSaver, ZarrDataset

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config_devel.json')

def evaluate_and_log(dataset, logger, metric_handler, dataset_name, distributed=False, comm=None):
    
    if comm.Get_rank() == 0:
        metrics = metric_handler(dataset[0])
        sample_out = {**metrics, "base_time": dataset[0]["base_time"], "lead_time": dataset[0]["lead_time"]}
        logger.initialize_store(sample_out)
    comm.Barrier()
    
    dataloader, sampler = get_dataloader(dataset, distributed=distributed)
    count = 0
    with torch.no_grad():
        for sample in dataloader:
            metrics = metric_handler(sample)
            sample_out = {**metrics, "base_time": sample["base_time"], "lead_time": sample["lead_time"]}
            logger.save(sample_out)
            count += 1
    print(f"Rank {comm.Get_rank()} processed {count} samples.")

def setup(distributed=False):
    if distributed:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']

        backend = "nccl" if torch.cuda.is_available() else "gloo"

        if rank==0:
            print('partition:\t', os.getenv('SLURM_JOB_PARTITION'))
            print('backend:\t', backend)

        dist.init_process_group(
            backend=backend,
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank
        )
        # print(f"Process group initialized for rank {rank} of {world_size} on CPU.")
    else:
        rank = 0
        world_size = 1

    return rank, world_size

def get_dataloader(dataset, distributed=False):
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0 if torch.cuda.is_available() else num_workers,
        # num_workers=0,  # Set to 0 for debugging or single-process mode
    )
    return dataloader, None


# NEW: Utility for constructing and extracting dataset metadata (used only on rank 0)
def build_dataset_info(config_path, dataset_key="model", shared_valid_times=None):
    dataset = UnifiedDataset(config_path, dataset_key, shared_valid_times=shared_valid_times)
    return {
        "samples": dataset.samples,
        "grid": dataset.grid,
        "metrics": dataset.metrics,
        "requested_names": dataset.requested_names,
        "canonical_names": dataset.canonical_names,
        "chosen_valid_times": dataset.chosen_valid_times
    }

def harmonize_zarr_to_xarray(
    zarr_group: zarr.hierarchy.Group,
    main_coord_name: str = 'base_time'
) -> xr.Dataset:
    """
    Builds a consistent xarray.Dataset from a Zarr group by harmonizing
    all variables and coordinates to a consistent size along shared dimensions.

    Args:
        zarr_group: An open Zarr group object (from zarr.open).
        main_coord_name: The name of the array to use as the primary
                         coordinate and reference for sizing the 'sample' dimension.

    Returns:
        A new, internally consistent xarray.Dataset object.
    """
    print(f"Starting robust harmonization based on '{main_coord_name}'...")

    try:
        main_coord_data = zarr_group[main_coord_name][:]
        target_sample_size = len(main_coord_data)
        sample_dim_name = 'idx'
    except KeyError:
        raise KeyError(f"Main coordinate '{main_coord_name}' not found.")

    # Define only the core, non-harmonized coordinates here
    coords = {
        sample_dim_name: (sample_dim_name, main_coord_data),
        'lat': ('lat', zarr_group['lat'][:]),
        'lon': ('lon', zarr_group['lon'][:]),
        # 'base_time': ()
    }
    
    # --- CHANGE ---
    # 'lead_time' is no longer treated as a special case here.
    # It will be processed in the main loop like any other variable.

    data_vars = {}
    # Process all keys except the ones we manually defined as coordinates
    vars_to_process = zarr_group.keys() - coords.keys()

    for key in vars_to_process:
        source_array = zarr_group[key]
        # Skip empty or non-array elements
        if not hasattr(source_array, 'shape'):
            continue
            
        data = source_array[:]

        # Infer dimension names based on shape
        dims = None
        if data.ndim >= 1 and source_array.name.lstrip('/') != main_coord_name:
            dims = [sample_dim_name] + [f'dim_{i}' for i in range(1, data.ndim)]
            if data.ndim == 4:
                if data.shape[0]==1:
                    print(f'dim_{key}')
                    dims = (sample_dim_name, f'dim_{key}', 'lat', 'lon')
                else:
                    dims = (sample_dim_name, 'level', 'lat', 'lon')
        else:
            continue # Skip main coordinate as it's already handled

        # --- Harmonize 'sample' dimension (axis 0) ---
        if data.shape[0] != target_sample_size:
            if data.shape[0] > target_sample_size:
                print(f"  -> Slicing '{key}' on '{sample_dim_name}': {data.shape[0]} -> {target_sample_size}")
                data = data[0:target_sample_size]
            else:
                print(f"  -> Padding '{key}' on '{sample_dim_name}': {data.shape[0]} -> {target_sample_size}")
                padding = [(0, target_sample_size - data.shape[0])] + [(0, 0)] * (data.ndim - 1)
                data = np.pad(data, padding, mode='constant', constant_values=np.nan)

        # --- Harmonize 'level' dimension (axis 1) ---
        if key == 'hydrostatic_rmse' and data.ndim == 4 and data.shape[1] == 1:
            print(f"  -> Broadcasting '{key}' on 'level' dim: 1 -> 3")
            data = np.repeat(data, 3, axis=1)

        data_vars[key] = xr.DataArray(data=data, dims=dims, name=key)

    ds_cleaned = xr.Dataset(data_vars, coords=coords)
    print("Harmonization complete.")
    return ds_cleaned
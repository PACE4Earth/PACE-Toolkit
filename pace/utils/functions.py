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

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config_devel.json')

def get_final_dataset(outputs_dir, model_name):

    try:
        # final_dataset = xr.open_zarr(os.path.join(outputs_dir, f"{model_name}.zarr"), consolidated=False)
        
        tmp_dataset = zarr.open(os.path.join(outputs_dir, f"{model_name}.zarr"), mode='r')
        
        print()
        print(tmp_dataset['idx'])
        try:
            print(tmp_dataset['idx'][:])
        except:
            print('hell naw')
        
        print(tmp_dataset.tree())
        
        # dirty bit
        
        # final_dataset = harmonize_zarr_to_xarray(tmp_dataset)
        final_dataset = harmonize_zarr_to_xarray(tmp_dataset)
        
        print()
        
        # try:
        #     print(final_dataset.tree())
        # except:
        #     print(final_dataset)
            
    except Exception as e:
        print(e)
        
    return final_dataset
    

def evaluate_and_log(dataset, logger, metric_handler, dataset_name, distributed=False, comm=None):
    
    if comm.Get_rank() == 0:
        metrics = metric_handler(dataset[0])
        sample_out = {**metrics, "base_time": dataset[0]["base_time"], "lead_time": dataset[0]["lead_time"], "idx": dataset[0]["idx"]}
        logger.initialize_store(sample_out)
    comm.Barrier()
    
    dataloader, sampler = get_dataloader(dataset, distributed=distributed)
    count = 0
    with torch.no_grad():
        for sample in dataloader:
            metrics = metric_handler(sample)
            sample_out = {**metrics, "base_time": sample["base_time"], "lead_time": sample["lead_time"], "idx": sample["idx"]}
            logger.save(sample_out)
            count += 1
    print(f"Rank {comm.Get_rank()} processed {count} samples.")

def setup(comm, distributed=False):
    if distributed:
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        # rank = int(os.environ['SLURM_PROCID'])
        # world_size = int(os.environ['SLURM_NTASKS'])
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
        rank = comm.Get_rank()
        world_size = comm.Get_size()
        
    if torch.cuda.is_available():
        logical_gpu_id = torch.cuda.current_device() 
        gpu_name = torch.cuda.get_device_name(logical_gpu_id)
        
        x = torch.ones(512*rank, device=logical_gpu_id)
        tensor_mem = x.nelement() * x.element_size() 
        allocated_mem = torch.cuda.memory_allocated()
        
        device = f'cuda:{logical_gpu_id}'
        
        try:
            x = x.to(device)
            print(x.device)
            print(allocated_mem)
        except Exception as e:
            print(e)
            
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
def build_dataset_info(config_path, dataset_key="model", shared_valid_times=None,):
    dataset = UnifiedDataset(config_path, dataset_key, shared_valid_times=shared_valid_times)
    return {
        "samples": dataset.samples,
        "grid": dataset.grid,
        "metrics": dataset.metrics,
        "requested_names": dataset.requested_names,
        "canonical_names": dataset.canonical_names,
        "chosen_valid_times": dataset.chosen_valid_times,
        "index_map": dataset.index_map
    }

def harmonize_zarr_to_xarray(
    zarr_group: zarr.hierarchy.Group,
    main_coord_name: str = 'idx'
) -> xr.Dataset:
    """
    Builds a consistent xarray.Dataset from a Zarr group by universally
    harmonizing all variables and coordinates along shared dimensions.
    """
    print(f"Starting universal harmonization based on '{main_coord_name}'...")

    try:
        main_coord_data = zarr_group[main_coord_name][:]
        target_sample_size = len(main_coord_data)
        sample_dim_name = 'idx'
    except KeyError:
        raise KeyError(f"Main coordinate '{main_coord_name}' not found.")

    coords = {
        sample_dim_name: (sample_dim_name, main_coord_data),
        'lat': ('lat', zarr_group['lat'][:]),
        'lon': ('lon', zarr_group['lon'][:]),
        'level': ('level', zarr_group['level'][:]),
        'base_time': (sample_dim_name, zarr_group['base_time'][:]),
        'lead_time': (sample_dim_name, zarr_group['lead_time'][:]),
    }

    try:
        target_level_size = len(coords['level'][1])
        target_lat_size = len(coords['lat'][1])
        target_lon_size = len(coords['lon'][1])
    except KeyError as e:
        raise KeyError(f"A required coordinate is missing: {e}")

    data_vars = {}
    vars_to_process = zarr_group.keys() - coords.keys()

    for key in vars_to_process:
        source_array = zarr_group[key]
        if not hasattr(source_array, 'shape') or source_array.shape == ():
            continue
        data = source_array[:]

        # --- NEW: Intelligent Dimension Naming ---
        dims = None
        if data.ndim >= 1:
            dims = [sample_dim_name]
            if data.ndim == 4:
                dims.extend(['level', 'dim_2', 'dim_3'])
                if data.shape[2] == target_lat_size:
                    dims[2] = 'lat'
                if data.shape[3] == target_lon_size:
                    dims[3] = 'lon'
            else:
                dims.extend([f'dim_{i}' for i in range(1, data.ndim)])
            dims = tuple(dims)
        else:
            continue

        # --- Harmonize 'sample' dimension (axis 0) ---
        if data.shape[0] != target_sample_size:
            # This logic remains the same...
            if data.shape[0] > target_sample_size:
                print(f"  -> Slicing '{key}' on '{sample_dim_name}': {data.shape[0]} -> {target_sample_size}")
                data = data[0:target_sample_size]
            else:
                print(f"  -> Padding '{key}' on '{sample_dim_name}': {data.shape[0]} -> {target_sample_size}")
                padding = [(0, target_sample_size - data.shape[0])] + [(0, 0)] * (data.ndim - 1)
                data = np.pad(data, padding, mode='constant', constant_values=np.nan)

        # --- UNIVERSAL Harmonization for 'level' dimension (axis 1) ---
        if 'level' in dims and data.shape[1] != target_level_size:
            current_level_size = data.shape[1]
            if current_level_size == 1:
                print(f"  -> Broadcasting '{key}' on 'level' dim: 1 -> {target_level_size}")
                data = np.repeat(data, target_level_size, axis=1)
            elif current_level_size < target_level_size:
                print(f"  -> Padding '{key}' on 'level' dim: {current_level_size} -> {target_level_size}")
                amount_to_pad = target_level_size - current_level_size
                pad_width = [(0, 0)] * data.ndim
                pad_width[1] = (0, amount_to_pad)
                data = np.pad(data, pad_width, mode='constant', constant_values=np.nan)
            else:
                print(f"  -> Slicing '{key}' on 'level' dim: {current_level_size} -> {target_level_size}")
                data = data[:, :target_level_size, ...]

        data_vars[key] = xr.DataArray(data=data, dims=dims, name=key)

    ds_cleaned = xr.Dataset(data_vars, coords=coords)
    print("Harmonization complete.")
    return ds_cleaned

# def harmonize_zarr_to_xarray(
#     zarr_group: zarr.hierarchy.Group,
#     main_coord_name: str = 'idx'
# ) -> xr.Dataset:
#     """
#     Builds a consistent xarray.Dataset from a Zarr group by harmonizing
#     all variables and coordinates to a consistent size along shared dimensions.

#     Args:
#         zarr_group: An open Zarr group object (from zarr.open).
#         main_coord_name: The name of the array to use as the primary
#                          coordinate and reference for sizing the 'sample' dimension.

#     Returns:
#         A new, internally consistent xarray.Dataset object.
#     """
#     print(f"Starting robust harmonization based on '{main_coord_name}'...")

#     try:
#         main_coord_data = zarr_group[main_coord_name][:]
#         target_sample_size = len(main_coord_data)
#         sample_dim_name = 'idx'
#     except KeyError:
#         raise KeyError(f"Main coordinate '{main_coord_name}' not found.")

#     # Define only the core, non-harmonized coordinates here
#     coords = {
#         sample_dim_name: (sample_dim_name, main_coord_data),
#         'lat': ('lat', zarr_group['lat'][:]),
#         'lon': ('lon', zarr_group['lon'][:]),
#         # 'base_time': ()
#     }
    
#     # --- CHANGE ---
#     # 'lead_time' is no longer treated as a special case here.
#     # It will be processed in the main loop like any other variable.

#     data_vars = {}
#     # Process all keys except the ones we manually defined as coordinates
#     vars_to_process = zarr_group.keys() - coords.keys()

#     for key in vars_to_process:
#         source_array = zarr_group[key]
#         # Skip empty or non-array elements
#         if not hasattr(source_array, 'shape'):
#             continue
            
#         data = source_array[:]

#         # Infer dimension names based on shape
#         dims = None
#         if data.ndim >= 1 and source_array.name.lstrip('/') != main_coord_name:
#             dims = [sample_dim_name] + [f'dim_{i}' for i in range(1, data.ndim)]
#             if data.ndim == 4:
#                 if data.shape[0]==1:
#                     print(f'dim_{key}')
#                     dims = (sample_dim_name, f'dim_{key}', 'lat', 'lon')
#                 else:
#                     dims = (sample_dim_name, 'level', 'lat', 'lon')
#         else:
#             continue # Skip main coordinate as it's already handled

#         # --- Harmonize 'sample' dimension (axis 0) ---
#         if data.shape[0] != target_sample_size:
#             if data.shape[0] > target_sample_size:
#                 print(f"  -> Slicing '{key}' on '{sample_dim_name}': {data.shape[0]} -> {target_sample_size}")
#                 data = data[0:target_sample_size]
#             else:
#                 print(f"  -> Padding '{key}' on '{sample_dim_name}': {data.shape[0]} -> {target_sample_size}")
#                 padding = [(0, target_sample_size - data.shape[0])] + [(0, 0)] * (data.ndim - 1)
#                 data = np.pad(data, padding, mode='constant', constant_values=np.nan)

#         # --- Harmonize 'level' dimension (axis 1) ---
#         if key == 'hydrostatic_rmse' and data.ndim == 4 and data.shape[1] == 1:
#             print(f"  -> Broadcasting '{key}' on 'level' dim: 1 -> 3")
#             data = np.repeat(data, 3, axis=1)

#         data_vars[key] = xr.DataArray(data=data, dims=dims, name=key)

#     ds_cleaned = xr.Dataset(data_vars, coords=coords)
#     print("Harmonization complete.")
#     return ds_cleaned
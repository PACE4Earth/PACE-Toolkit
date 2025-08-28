import os
from mpi4py import MPI

import numpy as np
import xarray as xr
import zarr

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from utils.dataset import UnifiedDataset

def evaluate_and_log(dataset, logger, metric_handler, dataset_name, distributed=False, comm=None):
    """
    Evaluates a dataset using a MetricHandler and logs the results.

    Args:
        dataset: PyTorch Dataset or list of samples.
        logger: Logger object that handles saving outputs.
        metric_handler: MetricHandler object with metrics to compute.
        dataset_name: Name of the dataset for logging purposes.
        distributed: If True, uses distributed dataloader.
        comm: MPI communicator. Defaults to MPI.COMM_WORLD.
    """
    if comm == None:
        comm = MPI.COMM_WORLD
    
    # Initialize logging on rank 0
    if comm.Get_rank() == 0:
        metrics = metric_handler(dataset[0])
        sample_out = {**metrics, "base_time": dataset[0]["base_time"], "lead_time": dataset[0]["lead_time"], "idx": dataset[0]["idx"]}
        logger.initialize_store(sample_out)
    comm.Barrier()
    
    dataloader, _ = get_dataloader(dataset, distributed=distributed)
    
    count = 0
    with torch.no_grad():
        for sample in dataloader:
            metrics = metric_handler(sample)
            sample_out = {**metrics, "base_time": sample["base_time"], "lead_time": sample["lead_time"], "idx": sample["idx"]}
            logger.save(sample_out)
            count += 1
            
        evaluate_accumulated(
            logger=logger,
            metric_handler=metric_handler,
            dataset_name=dataset_name,
            comm=comm,
        )
            
    print(f"Rank {comm.Get_rank()} processed {count} samples.")

def evaluate_accumulated(logger, metric_handler, dataset_name, comm):
    """
    Evaluates all accumulated metrics using the logger.

    Args:
        logger: Logger object storing metric outputs.
        metric_handler: MetricHandler with registered metrics.
        dataset_name: Name of the dataset.
        comm: MPI communicator.
    """
    for metric, module in metric_handler.metrics.items():
        try:
            module.evaluate(logger, comm)
            # print(metric, 'success')
        except Exception as e:
            if comm.Get_rank() == 0:
                print(metric, e)
                pass
    
    return None

def setup(comm, distributed=False):
    """
    Initializes MPI and optionally PyTorch distributed environment.

    Args:
        comm: MPI communicator.
        distributed: If True, sets up PyTorch distributed.

    Returns:
        rank: MPI rank of current process.
        world_size: Total number of MPI processes.
    """
    if distributed:
        rank = comm.Get_rank()
        world_size = comm.Get_size()
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
       
    comm.Barrier()
         
    return rank, world_size


def get_dataloader(dataset, distributed=False):
    """
    Returns a PyTorch DataLoader for a dataset.

    Args:
        dataset: PyTorch Dataset object.
        distributed: If True, uses DistributedSampler (not implemented here).

    Returns:
        dataloader: PyTorch DataLoader object.
        sampler: Currently None, placeholder for future distributed support.
    """
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0 if torch.cuda.is_available() else num_workers,
        # num_workers=0,  # Set to 0 for debugging or single-process mode
    )
    return dataloader, None


def build_dataset_info(config_path, dataset_key="model", shared_valid_times=None,):
    """
    Constructs a UnifiedDataset and extracts its metadata.

    Args:
        config_path: Path to the dataset configuration JSON.
        dataset_key: Key identifying the dataset in the config.
        shared_valid_times: Optional list of valid_times to restrict dataset.

    Returns:
        Dictionary containing samples, grid, metrics, and other metadata.
    """
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

# def harmonize_zarr_to_xarray(
#     zarr_group: zarr.hierarchy.Group,
#     main_coord_name: str = 'idx'
# ) -> xr.Dataset:
#     """
#     Builds a consistent xarray.Dataset from a Zarr group by universally
#     harmonizing all variables and coordinates along shared dimensions.
#     """
#     print(f"Starting universal harmonization based on '{main_coord_name}'...")

#     try:
#         main_coord_data = zarr_group[main_coord_name][:]
#         target_sample_size = len(main_coord_data)
#         sample_dim_name = 'idx'
#     except KeyError:
#         raise KeyError(f"Main coordinate '{main_coord_name}' not found.")

#     coords = {
#         sample_dim_name: (sample_dim_name, main_coord_data),
#         'lat': ('lat', zarr_group['lat'][:]),
#         'lon': ('lon', zarr_group['lon'][:]),
#         'level': ('level', zarr_group['level'][:]),
#         'base_time': (sample_dim_name, zarr_group['base_time'][:]),
#         'lead_time': (sample_dim_name, zarr_group['lead_time'][:]),
#     }

#     try:
#         target_level_size = len(coords['level'][1])
#         target_lat_size = len(coords['lat'][1])
#         target_lon_size = len(coords['lon'][1])
#     except KeyError as e:
#         raise KeyError(f"A required coordinate is missing: {e}")

#     data_vars = {}
#     vars_to_process = zarr_group.keys() - coords.keys()

#     for key in vars_to_process:
#         source_array = zarr_group[key]
#         if not hasattr(source_array, 'shape') or source_array.shape == ():
#             continue
#         data = source_array[:]

#         # --- NEW: Intelligent Dimension Naming ---
#         dims = None
#         if data.ndim >= 1:
#             dims = [sample_dim_name]
#             if data.ndim == 4:
#                 dims.extend(['level', 'var_2', 'var_3'])
#                 if data.shape[2] == target_lat_size:
#                     dims[2] = 'lat'
#                 if data.shape[3] == target_lon_size:
#                     dims[3] = 'lon'
#             else:
#                 dims.extend([f'var_{i}' for i in range(1, data.ndim)])
#             dims = tuple(dims)
#         else:
#             continue

#         # --- Harmonize 'sample' dimension (axis 0) ---
#         if data.shape[0] != target_sample_size:
#             # This logic remains the same...
#             if data.shape[0] > target_sample_size:
#                 print(f"  -> Slicing '{key}' on '{sample_dim_name}': {data.shape[0]} -> {target_sample_size}")
#                 data = data[0:target_sample_size]
#             else:
#                 print(f"  -> Padding '{key}' on '{sample_dim_name}': {data.shape[0]} -> {target_sample_size}")
#                 padding = [(0, target_sample_size - data.shape[0])] + [(0, 0)] * (data.ndim - 1)
#                 data = np.pad(data, padding, mode='constant', constant_values=np.nan)

#         # --- UNIVERSAL Harmonization for 'level' dimension (axis 1) ---
#         if 'level' in dims and data.shape[1] != target_level_size:
#             current_level_size = data.shape[1]
#             if current_level_size == 1:
#                 print(f"  -> Broadcasting '{key}' on 'level' dim: 1 -> {target_level_size}")
#                 data = np.repeat(data, target_level_size, axis=1)
#             elif current_level_size < target_level_size:
#                 print(f"  -> Padding '{key}' on 'level' dim: {current_level_size} -> {target_level_size}")
#                 amount_to_pad = target_level_size - current_level_size
#                 pad_width = [(0, 0)] * data.ndim
#                 pad_width[1] = (0, amount_to_pad)
#                 data = np.pad(data, pad_width, mode='constant', constant_values=np.nan)
#             else:
#                 print(f"  -> Slicing '{key}' on 'level' dim: {current_level_size} -> {target_level_size}")
#                 data = data[:, :target_level_size, ...]

#         data_vars[key] = xr.DataArray(data=data, dims=dims, name=key)

#     ds_cleaned = xr.Dataset(data_vars, coords=coords)
#     print("Harmonization complete.")
#     return ds_cleaned
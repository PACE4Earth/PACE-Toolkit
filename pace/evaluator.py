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
from metrics.metric_handler import MetricHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config_devel.json')

def setup(distributed=False):
    if distributed:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']

        dist.init_process_group(
            backend="gloo",
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
        num_workers=num_workers,
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
    Builds a consistent xarray.Dataset from an open but inconsistent Zarr group.

    It harmonizes variables by slicing or padding them along their first
    dimension to match the length of a specified main coordinate.

    Args:
        zarr_group: An open Zarr group object (from zarr.open).
        main_coord_name: The name of the 1D array to use as the primary
                         coordinate and reference for sizing.

    Returns:
        A new, internally consistent xarray.Dataset object.
    """
    print(f"Robustly harmonizing Zarr group based on '{main_coord_name}'...")

    try:
        main_coord_data = zarr_group[main_coord_name][:]
        target_size = len(main_coord_data)
        main_dim_name = 'sample'
    except KeyError:
        raise KeyError(f"Main coordinate '{main_coord_name}' not found.")

    coords = {
        main_dim_name: (main_dim_name, main_coord_data),
        'y': ('y', zarr_group['lat'][:]),
        'x': ('x', zarr_group['lon'][:]),
    }
    if 'lead_time' in zarr_group:
         coords['lead_time'] = (main_dim_name, zarr_group['lead_time'][0:target_size])


    data_vars = {}
    vars_to_process = zarr_group.keys() - coords.keys() - {main_coord_name}

    for key in vars_to_process:
        source_array = zarr_group[key]
        data = source_array[:]
        
        dims = None
        if data.ndim == 4:
            dims = (main_dim_name, 'level', 'y', 'x')
        elif data.ndim >= 1: # Handle other dimensionalities
            # Use generic dim names for simplicity
            dims = [main_dim_name] + [f'dim_{i}' for i in range(1, data.ndim)]
        else:
            continue

        # --- Improved Harmonization Logic ---
        if data.shape[0] != target_size:
            # Case 1: Array is too large, slice it down.
            if data.shape[0] > target_size:
                print(f"  -> Slicing '{key}': {data.shape[0]} -> {target_size}")
                data = data[0:target_size]
            # Case 2: Array is too small, pad it with NaNs.
            else:
                print(f"  -> Padding '{key}': {data.shape[0]} -> {target_size}")
                pad_width = target_size - data.shape[0]
                # Create a tuple of pad widths for each dimension ((before, after), ...)
                padding = [(0, pad_width)] + [(0, 0)] * (data.ndim - 1)
                data = np.pad(data, padding, mode='constant', constant_values=np.nan)

        data_vars[key] = xr.DataArray(data=data, dims=dims, name=key)

    ds_cleaned = xr.Dataset(data_vars, coords=coords)
    print("Harmonization complete.")
    return ds_cleaned

def main(distributed=False):
    time_start = time.perf_counter()
    rank, world_size = setup(distributed=distributed)
    comm = MPI.COMM_WORLD
    assert world_size == comm.Get_size()
    assert rank == comm.Get_rank()

    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    outputs_dir = os.path.expandvars(config.get("outputs_dir", os.path.join(BASE_DIR, "outputs")))
    os.makedirs(outputs_dir, exist_ok=True)

    # print('output dir:', outputs_dir)

    # RANK 0 builds the full dataset and sample list
    if rank == 0:
        model_info = build_dataset_info(DATASET_CONFIG_PATH, dataset_key="model")
    else:
        model_info = None

    model_info = comm.bcast(model_info, root=0)
    rank_samples = model_info["samples"][rank::world_size]

    model_dataset = UnifiedDataset.from_sample_list(
        sample_list=rank_samples,
        grid=model_info["grid"],
        metrics=model_info["metrics"],
        requested_names=model_info["requested_names"],
        canonical_names=model_info["canonical_names"],
        config_path=DATASET_CONFIG_PATH,
        dataset_key="model"
    )

    # Repeat the same for reference dataset, if present
    if "reference" in config.get("datasets", {}):
        if rank == 0:
            ref_info = build_dataset_info(
                DATASET_CONFIG_PATH, dataset_key="reference",
                shared_valid_times=model_info["chosen_valid_times"]
            )
        else:
            ref_info = None

        ref_info = comm.bcast(ref_info, root=0)
        ref_rank_samples = ref_info["samples"][rank::world_size]
        reference_dataset = UnifiedDataset.from_sample_list(
            sample_list=ref_rank_samples,
            grid=ref_info["grid"],
            metrics=ref_info["metrics"],
            requested_names=ref_info["requested_names"],
            canonical_names=ref_info["canonical_names"],
            config_path=DATASET_CONFIG_PATH,
            dataset_key="reference"
        )
    else:
        reference_dataset = None

    model_name = config["datasets"]["model"]["name"]
    reference_name = config["datasets"].get("reference", {}).get("name")

    model_output_logger = MPIZarrSaver(
        path=os.path.join(outputs_dir, f"{model_name}.zarr"), 
        comm=comm,
        lat=model_dataset.grid['lat'],
        lon=model_dataset.grid['lon'],
    )

    # Save static coordinates once (only rank 0)
    # if rank == 0:
    #     coords_to_save = {}
    #     for coord_name in ["lat", "lon", "pressure_levels"]:
    #         if coord_name in model_info["grid"]:
    #             coords_to_save[coord_name] = np.array(model_info["grid"][coord_name])
    #     # Save to Zarr root group
    #     zarr_path = os.path.join(outputs_dir, f"{model_name}.zarr")
    #     root = zarr.open(zarr_path, mode="a")
    #     for k, v in coords_to_save.items():
    #         if k not in root:
    #             root.create_dataset(k, data=v, overwrite=True)
    if reference_dataset:
        reference_output_logger = MPIZarrSaver(
            path=os.path.join(outputs_dir, f"{reference_name}.zarr"),
            comm=comm,
        )

        # if rank == 0:
        #     coords_to_save = {}
        #     for coord_name in ["lat", "lon", "pressure_levels"]:
        #         if coord_name in ref_info["grid"]:
        #             coords_to_save[coord_name] = np.array(ref_info["grid"][coord_name])
        #     zarr_path = os.path.join(outputs_dir, f"{reference_name}.zarr")
        #     root = zarr.open(zarr_path, mode="a")
        #     for k, v in coords_to_save.items():
        #         if k not in root:
        #             root.create_dataset(k, data=v, overwrite=True)

    metric_handler = MetricHandler(
        metrics=list(model_dataset.metrics.keys()),
        grid=model_dataset.grid
    )
    
    

    def evaluate_and_log(dataset, logger, dataset_name):
        
        if rank==0:
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
        print(f"Rank {rank} processed {count} samples.")


    evaluate_and_log(model_dataset, model_output_logger, dataset_name=model_name)

    if reference_dataset:
        
        if rank==0:
            reference_output_logger.initialize_store(reference_dataset[0])

        comm.Barrier()

        evaluate_and_log(reference_dataset, reference_output_logger, dataset_name=reference_name)

    time.sleep(0.1)

    comm.Barrier()

    if distributed:
        dist.destroy_process_group()

    print(f"Rank {comm.Get_rank()} waiting at barrier.")
    comm.Barrier()
    print(f"Rank {comm.Get_rank()} passed barrier.")

    if comm.Get_rank() == 0:
        print("\n--- All ranks finished writing. Now performing final check. ---")
        
        
        try:
            # final_dataset = xr.open_zarr(os.path.join(outputs_dir, f"{model_name}.zarr"), consolidated=False)
            tmp_dataset = zarr.open(os.path.join(outputs_dir, f"{model_name}.zarr"), mode='r')
            print(tmp_dataset.tree())
            final_dataset = harmonize_zarr_to_xarray(tmp_dataset)
            try:
                print(final_dataset.tree())
            except:
                print(final_dataset)
            # print(final_dataset)
        except Exception as e:
            print(e)
    
        time_end = time.perf_counter()
        print(f"Elapsed time: {time_end - time_start:.2f} s")

if __name__ == "__main__":
    main(distributed=True)

import os
import shutil
import json
from mpi4py import MPI
from collections import defaultdict
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

import xarray.backends.zarr
from xarray.core.utils import is_dict_like

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from utils.dataset import UnifiedDataset
from utils.output_logger import MPIZarrSaver
from metrics.metric_handler import MetricHandler
from utils.functions import (
    setup,
    # get_dataloader,
    build_dataset_info,
    # harmonize_zarr_to_xarray,
    evaluate_and_log,
    get_final_dataset,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CONFIG_PATH_DEFAULT = os.path.join(BASE_DIR, 'configs', 'config_graphcast.json')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU index: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

os.environ['DEVICE'] = DEVICE

def main(distributed=False):
        
    time_start = time.perf_counter()
    
    try:
        DATASET_CONFIG_PATH = Path(os.environ['DATASET_CONFIG_PATH'])
    except:
        DATASET_CONFIG_PATH = DATASET_CONFIG_PATH_DEFAULT
        
    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)
            
    
    distributed = config.get("distributed", distributed)
    
    try:
        outputs_dir = os.environ['OUTPUT_DIR_PATH']
    except:
        outputs_dir = config.get("outputs_dir", None)
    
    
    comm = MPI.COMM_WORLD
    
    
    rank, world_size = setup(comm=comm, distributed=distributed)
    if rank == 0:
        print('ds_config_path:', DATASET_CONFIG_PATH)
        print('output dir:', outputs_dir)
        print(f"DEBUG: rank={rank}, world_size={world_size}, comm_size={comm.Get_size()}, comm_rank={comm.Get_rank()}")
    # assert world_size == comm.Get_size()
    # assert rank == comm.Get_rank()

    try:
        if rank==0:
            torch.multiprocessing.set_start_method('spawn')
    except:
        ...
    comm.Barrier()
    

    # RANK 0 builds the full dataset and sample list
    if rank == 0:
        model_info = build_dataset_info(DATASET_CONFIG_PATH, dataset_key="model")
    else:
        model_info = None

    model_info = comm.bcast(model_info, root=0)
    len_samples = len(model_info["samples"])
    print(len_samples)
    # Split indices for each rank
    # all_indices = np.arange(len_samples)
    # rank_indices = all_indices[rank*len_samples//world_size:(rank+1)*len_samples//world_size]
    # rank_samples = [model_info["samples"][i] for i in rank_indices]
    rank_samples = model_info["samples"][rank::world_size]
    
    # print(f"Rank {rank} received {len(rank_samples)} samples, indices: {rank_indices}")

    model_dataset = UnifiedDataset.from_sample_list(
        sample_list=rank_samples,
        grid=model_info["grid"],
        metrics=model_info["metrics"],
        requested_names=model_info["requested_names"],
        canonical_names=model_info["canonical_names"],
        config_path=DATASET_CONFIG_PATH,
        dataset_key="model",
        index_map=model_info["index_map"]
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
            dataset_key="reference",
            index_map=ref_info["index_map"]
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
        level=model_dataset.grid['pressure_levels'],
        N_total=len(model_info["samples"])
    )

    if reference_dataset:
        reference_output_logger = MPIZarrSaver(
            path=os.path.join(outputs_dir, f"{reference_name}.zarr"),
            comm=comm,
            lat=model_dataset.grid['lat'],
            lon=model_dataset.grid['lon'],
            level=model_dataset.grid['pressure_levels'],
            N_total=len(ref_info["samples"])
        )

    metric_handler = MetricHandler(
        metrics=list(model_dataset.metrics.keys()),
        grid=model_dataset.grid,
        config_path=DATASET_CONFIG_PATH,
    )

    comm.Barrier()

    evaluate_and_log(
        dataset=model_dataset, 
        logger=model_output_logger, 
        dataset_name=model_name, 
        metric_handler=metric_handler, 
        distributed=distributed,
        comm=comm,
    )

    print(f"Rank {comm.Get_rank()} waiting at barrier.")
    comm.Barrier()
    print(f"Rank {comm.Get_rank()} passed barrier.")
    
    comm.Barrier()
    if rank==0:
        print(f'Passed barrier after {model_name}.')    
    time.sleep(0.1)

    if reference_dataset:
        evaluate_and_log(
            dataset=reference_dataset, 
            logger=reference_output_logger, 
            dataset_name=reference_name,
            metric_handler=metric_handler, 
            distributed=distributed,
            comm=comm,
        )
        comm.Barrier()


    print(f"Rank {comm.Get_rank()} waiting at barrier.")
    comm.Barrier()
    if rank==0:
        print(f'Passed barrier after {reference_name}.')    
    time.sleep(0.1)

    comm.Barrier()

    if distributed:
        dist.destroy_process_group()
        
    comm.Barrier()
    
    model_store_path = os.path.join(outputs_dir, f"{model_name}.zarr")
    ref_store_path = os.path.join(outputs_dir, f"{reference_name}.zarr")

    if comm.Get_rank() == 0:
        print("Reconsolidating metadata...")
        # Make sure no stale lock file
        lockfile = os.path.join(model_store_path, '.zarrlock')
        if os.path.exists(lockfile):
            shutil.rmtree(lockfile)
        grp = zarr.open_group(model_store_path, mode="r")
        zarr.consolidate_metadata(model_store_path)

        if reference_dataset:
            lockfile = os.path.join(ref_store_path, '.zarrlock')
            if os.path.exists(lockfile):
                shutil.rmtree(lockfile)
            grp = zarr.open_group(ref_store_path, mode="r")
            zarr.consolidate_metadata(ref_store_path)

        print("Done.")

    comm.Barrier()

    if comm.Get_rank() == 0:
        
        # TODO:
        # correlation map <- accumulate, sum partials, evaluate
        # histograms/bivariate histograms -> sum, evaluate
        
        
        print("\n--- All ranks finished writing. Now performing final check. ---")
    
        tmp_dataset = zarr.open(os.path.join(outputs_dir, f"{model_name}.zarr"), mode='r')

        print(tmp_dataset['idx'])
        print(tmp_dataset['idx'][:])
        print(tmp_dataset.tree())
            
        try:
            final_dataset = xr.open_zarr(os.path.join(outputs_dir, f'{model_name}.zarr'))
            print('Opened using xarray.')
        except Exception as e:
            print(e)
            final_dataset = get_final_dataset(outputs_dir, model_name)
            
        print(final_dataset)
        
        time_end = time.perf_counter()
        print(f"Elapsed time: {time_end - time_start:.2f} s")


if __name__ == "__main__":
    main()

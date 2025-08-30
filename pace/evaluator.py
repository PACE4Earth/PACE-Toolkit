"""
Evaluator script.

This script evaluates model and optional reference datasets using configured metrics.
Supports distributed evaluation via MPI (and optionally PyTorch distributed).

Key features:
- Loads model and reference datasets from pre-built sample lists.
- Uses MetricHandler to compute metrics for each dataset.
- Logs outputs in Zarr format with MPIZarrSaver.
- Handles distributed execution across ranks and GPUs.
- Consolidates Zarr metadata and performs optional final checks.
"""

import os
import shutil
import json
import zarr
import time
from pathlib import Path
import argparse

import numpy as np
import xarray as xr
from mpi4py import MPI
import torch
import torch.distributed as dist

from utils.dataset import UnifiedDataset
from utils.output_logger import MPIZarrSaver
from metrics.metric_handler import MetricHandler
from utils.functions import setup, build_dataset_info, evaluate_and_log

# === GLOBAL DEVICE CONFIGURATION ===
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
os.environ['DEVICE'] = DEVICE

if torch.cuda.is_available():
    print(f"Number of GPUs available: {torch.cuda.device_count()}")
    print(f"Current GPU index: {torch.cuda.current_device()}")
    print(f"Current GPU name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


def main(distributed=False):
    """
    Main evaluation routine for the PACE project.

    Steps:
        1. Load configuration from file, environment variable, or default.
        2. Initialize MPI and optionally PyTorch distributed.
        3. Build datasets and rank-specific sample lists.
        4. Evaluate model and reference datasets using MetricHandler.
        5. Consolidate Zarr outputs and optionally perform final checks.

    Args:
        distributed (bool): Whether to enable distributed evaluation.
    """
    time_start = time.perf_counter()
    
    # === CONFIGURATION ===
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DEFAULT_CONFIG_PATH = Path(os.path.join(BASE_DIR, 'configs', 'config_template.json'))

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to JSON config file (overrides env var and default)")
    args = parser.parse_args()

    # Determine configuration path: arg > env var > default
    if args.config is not None:
        config_path = Path(args.config)
    elif "CONFIG_PATH" in os.environ:
        config_path = Path(os.environ["CONFIG_PATH"])
    else:
        config_path = DEFAULT_CONFIG_PATH

    config_path = config_path.expanduser().resolve()
    with open(config_path, "r") as f:
        config = json.load(f)

    distributed = config.get("distributed", distributed)
    outputs_dir = os.environ.get('OUTPUTS_DIR_PATH', config.get("outputs_dir", None))
    
    # === MPI / DISTRIBUTED SETUP ===
    comm = MPI.COMM_WORLD
    rank, world_size = setup(comm=comm, distributed=distributed)
    # if rank == 0:
    #     print('Configuration loaded from:', config_path)
    #     print('Output directory:', outputs_dir)
    #     print(f"DEBUG: rank={rank}, world_size={world_size}, comm_size={comm.Get_size()}, comm_rank={comm.Get_rank()}")
    #     print()
    
    # Ensure proper multiprocessing start method on rank 0
    try:
        if rank==0:
            torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass  # Already set; safe to ignore
    comm.Barrier()
    
    # === BUILD MODEL DATASET ===
    # RANK 0 builds the full dataset and sample list
    if rank == 0:
        model_info = build_dataset_info(config_path, dataset_key="model")
    else:
        model_info = None

    # Broadcast model_info to all ranks
    model_info = comm.bcast(model_info, root=0)

    # Assign rank-specific samples using strided slicing
    rank_samples = model_info["samples"][rank::world_size]
    
    # Instantiate UnifiedDataset for model
    model_dataset = UnifiedDataset.from_sample_list(
        sample_list=rank_samples,
        grid=model_info["grid"],
        metrics=model_info["metrics"],
        requested_names=model_info["requested_names"],
        canonical_names=model_info["canonical_names"],
        config_path=config_path,
        dataset_key="model",
        index_map=model_info["index_map"],
        name=model_info["name"]
    )

    # === BUILD REFERENCE DATASET IF AVAILABLE ===
    # Repeat the same for reference dataset
    if "reference" in config.get("datasets", {}):
        if rank == 0:
            ref_info = build_dataset_info(
                config_path, dataset_key="reference",
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
            config_path=config_path,
            dataset_key="reference",
            index_map=ref_info["index_map"],
            name=ref_info["name"]
        )
    else:
        reference_dataset = None

    # === SETUP OUTPUT LOGGERS ===
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

    # === METRIC HANDLER ===
    metric_handler = MetricHandler(
        metrics=list(model_dataset.metrics.keys()),
        grid=model_dataset.grid,
        config_path=config_path,
    )

    comm.Barrier()

    # === EVALUATE MODEL DATASET ===
    evaluate_and_log(
        dataset=model_dataset, 
        logger=model_output_logger, 
        dataset_name=model_name, 
        metric_handler=metric_handler, 
        distributed=distributed,
        comm=comm,
    )

    comm.Barrier()
    if rank == 0:
        print(f'Finished evaluation of {model_name}.\n')
    
    # === EVALUATE REFERENCE DATASET ===
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
        if rank == 0:
            print(f'Finished evaluation of {reference_name}.\n')

    # === CLEANUP DISTRIBUTED RESOURCES ===
    if distributed:
        dist.destroy_process_group()
    comm.Barrier()
    
    # === CONSOLIDATE ZARR METADATA ===
    if rank == 0:
        for ds_name in [model_name, reference_name] if reference_dataset else [model_name]:
            store_path = os.path.join(outputs_dir, f"{ds_name}.zarr")
            lockfile = os.path.join(store_path, '.zarrlock')
            if os.path.exists(lockfile):
                shutil.rmtree(lockfile)
            # Consolidate Zarr metadata for multi-rank access
            grp = zarr.open_group(store_path, mode="r")
            zarr.consolidate_metadata(store_path)
        print("Metadata consolidation complete.\n")

    comm.Barrier()

    # === OPTIONAL FINAL CHECKS ===
    if rank == 0:
        try:
            # Open final model dataset using xarray to verify correctness
            final_dataset = xr.open_zarr(os.path.join(outputs_dir, f'{model_name}.zarr'))
            print('Opening model dataset using xarray.\n')
            print(final_dataset)
        except Exception as e:
            print("Failed to open final dataset:", e)

        time_end = time.perf_counter()
        print(f"\nElapsed time: {time_end - time_start:.2f} s")


if __name__ == "__main__":
    main()

import os
import json
from mpi4py import MPI
from collections import defaultdict
import time

import numpy as np
import xarray as xr
import zarr

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from utils.dataset import UnifiedDataset
from utils.output_logger import MPIZarrSaver, ZarrHandler
from metrics.metric_handler import MetricHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'dataset_config.json')

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
        print(f"Process group initialized for rank {rank} of {world_size} on CPU.")
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

    print('output dir:', outputs_dir)

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
    )
    reference_output_logger = MPIZarrSaver(
        path=os.path.join(outputs_dir, f"{reference_name}.zarr"),
        comm=comm,
    ) if reference_dataset else None

    metric_handler = MetricHandler(
        metrics=list(model_dataset.metrics.keys()),
        grid=model_dataset.grid
    )

    def evaluate_and_log(dataset, logger, dataset_name):
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
        evaluate_and_log(reference_dataset, reference_output_logger, dataset_name=reference_name)

    comm.Barrier()

    if distributed:
        dist.destroy_process_group()

    print(f"Rank {comm.Get_rank()} waiting at barrier.")
    comm.Barrier()
    print(f"Rank {comm.Get_rank()} passed barrier.")

    if comm.Get_rank() == 0:
        print("\n--- All ranks finished writing. Now performing final check. ---")
        try:
            print("Consolidating Zarr metadata...")
            zarr.consolidate_metadata(os.path.join(outputs_dir, f"{model_name}.zarr"))
            if reference_name:
                zarr.consolidate_metadata(os.path.join(outputs_dir, f"{reference_name}.zarr"))
            print("Metadata consolidated.")
        except Exception as e:
            print(f"Could not consolidate metadata: {e}")

        print("Initializing a fresh reader object for final verification...")
        model_dataset = ZarrHandler(path=os.path.join(outputs_dir, f"{model_name}.zarr"), mode='r')
        print(f"Model dataset size: {len(model_dataset)}")

        if reference_name:
            ref_dataset = ZarrHandler(path=os.path.join(outputs_dir, f"{reference_name}.zarr"), mode='r')
            print(f"Reference dataset size: {len(ref_dataset)}")
    
        time_end = time.perf_counter()
        print(f"Elapsed time: {time_end - time_start:.2f} s")

if __name__ == "__main__":
    main(distributed=True)

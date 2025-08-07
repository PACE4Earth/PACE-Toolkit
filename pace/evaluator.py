import os
import json
from mpi4py import MPI
from collections import defaultdict

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
DATASET_CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'dataset_config_devel.json')

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
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,
    )
    return dataloader, sampler

def main(distributed=False):
    rank, world_size = setup(distributed=distributed)

    comm = MPI.COMM_WORLD
    
    # Optional sanity check: Ensure the worlds are consistent
    assert world_size == comm.Get_size(), "Mismatch between torch.distributed and MPI world sizes!"
    assert rank == comm.Get_rank(), "Mismatch between torch.distributed and MPI ranks!"

    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    outputs_dir = os.path.expandvars(config.get("outputs_dir", os.path.join(BASE_DIR, "outputs")))
    os.makedirs(outputs_dir, exist_ok=True)
    
    print('output dir:', outputs_dir)

    model_dataset = UnifiedDataset(
        DATASET_CONFIG_PATH, 
        dataset_key="model",
    )
    reference_dataset = UnifiedDataset(
        DATASET_CONFIG_PATH,
        dataset_key="reference",
        shared_valid_times=model_dataset.chosen_valid_times
    ) if "reference" in config.get("datasets", {}) else None

    model_name = config["datasets"]["model"]["name"]
    reference_name = config["datasets"].get("reference", {}).get("name")

    # model_output_logger = ZarrHandler(path=os.path.join(outputs_dir, f"{model_name}.zarr"))
    # reference_output_logger = ZarrHandler(path=os.path.join(outputs_dir, f"{reference_name}.zarr")) if reference_dataset else None

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
        if distributed:
            sampler.set_epoch(0)

        with torch.no_grad():
            for it, sample in enumerate(dataloader):
                metrics = metric_handler(sample)
                sample_out = {**metrics, "base_time": sample["base_time"], "lead_time": sample["lead_time"]}
                # logger(sample_out)
                logger.save(sample_out)

    # Evaluate model
    evaluate_and_log(model_dataset, model_output_logger, dataset_name=model_name)

    # Save reference outputs if available
    if reference_dataset:
        # def passthrough_logger(sample):
        #     sample_out = {
        #         name: tensor for name, tensor in sample.items()
        #         if name not in ["lat", "lon"] and torch.is_tensor(tensor)
        #     }
        #     sample_out["base_time"] = sample["base_time"]
        #     sample_out["lead_time"] = sample["lead_time"]
        #     reference_output_logger.save(sample_out)

        # evaluate_and_log(reference_dataset, passthrough_logger, dataset_name=reference_name)
        evaluate_and_log(reference_dataset, reference_output_logger, dataset_name=reference_name)

    comm.Barrier()

    if distributed:
        dist.destroy_process_group()

    print(f"Rank {comm.Get_rank()} waiting at barrier.")
    comm.Barrier()
    print(f"Rank {comm.Get_rank()} passed barrier.")

    # 2. On a single rank (e.g., Rank 0), perform final actions.
    if comm.Get_rank() == 0:
        print("\n--- All ranks finished writing. Now performing final check. ---")
        
        # Optional but recommended: Consolidate metadata for faster reads later.
        # This reads all the small `.zarray`, `.zgroup` files and puts them
        # into a single `.zmetadata` file.
        try:
            print("Consolidating Zarr metadata...")
            zarr.consolidate_metadata(os.path.join(outputs_dir, f"{model_name}.zarr"))
            print("Metadata consolidated.")
        except Exception as e:
            print(f"Could not consolidate metadata: {e}")

        # 3. Create a NEW reader instance AFTER the barrier.
        #    This guarantees it reads the final state from the disk.
        print("Initializing a fresh reader object for final verification...")
        final_dataset = ZarrHandler(path=os.path.join(outputs_dir, f"{model_name}.zarr"), mode='r')
        # 4. NOW the length will be correct. ✅
        print(f"Final dataset size: {len(final_dataset)}")
        
        if len(final_dataset) > 0:
            first_item = final_dataset[0]
            print(f"Successfully read first item with base_time: {first_item['base_time']}")


if __name__ == "__main__":
    main(distributed=True)

import os
import json
import torch
import torch.distributed as dist
from torch.utils.data import (
    Subset,
    DataLoader,
    DistributedSampler,
    RandomSampler,
)

from utils.dataset import UnifiedDataset
from metrics.metric_handler import MetricHandler

DATASET_CONFIG_PATH = 'configs/graphcast_extended.json'

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
        print('__________________________________________________')
        print(f'{master_addr} : {master_port}')
        print(f"Process group initialized for rank {rank} of {world_size} on CPU.")
        print('__________________________________________________')
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

def main(distributed=False, subset_length=None):
    rank, world_size = setup(distributed=distributed)

    # Load dataset config
    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Load the full dataset (UnifiedDataset instance)
    full_dataset = UnifiedDataset(DATASET_CONFIG_PATH)

    # Use a subset if requested, but keep the full_dataset reference
    if subset_length is not None:
        dataset = Subset(full_dataset, list(range(subset_length)))
    else:
        dataset = full_dataset

    # Prepare dataloader
    dataloader, sampler = get_dataloader(dataset=dataset, distributed=distributed)

    # Metric handler setup
    metric_handler = MetricHandler(
        metrics=list(full_dataset.metrics.keys()),
        grid=full_dataset.grid
    )

    # If using distributed evaluation
    if distributed:
        sampler.set_epoch(0)

    # Evaluation loop
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            if i == 0 and rank == 0:
                print(f"\n[Info] First batch sample keys: {list(sample.keys())}")
                for k in sample:
                    if isinstance(sample[k], torch.Tensor):
                        print(f"    {k:<25} -> shape {tuple(sample[k].shape)}")
                    else:
                        print(f"    {k:<25} -> {sample[k]}")
                print()

            # Optional: simple inspection
            if "geopotential" in sample:
                print(f"Rank {rank} Batch {i}, geopotential mean: {sample['geopotential'].mean().item():.4f}")
            elif "model_geopotential" in sample:
                print(f"Rank {rank} Batch {i}, model geopotential mean: {sample['model_geopotential'].mean().item():.4f}")

            # Run metrics
            output = metric_handler(sample)
            print(f"Rank {rank} Batch {i} -> metric keys: {list(output.keys())}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main(
        distributed=False,
        subset_length=20,
    )

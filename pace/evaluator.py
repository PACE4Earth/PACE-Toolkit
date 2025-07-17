import os
from torch.utils.data import (
    Subset, 
    DataLoader, 
    DistributedSampler, 
    RandomSampler,
)

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from utils.dataset import UnifiedDataset
from metrics.metric_handler import MetricHandler

DATASET_CONFIG_PATH = './pace/configs/graphcast_extended.json'

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
        print(f'{master_addr} : {master_port}')
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

def main(
    distributed=False, 
    subset_length=None,
    ):
    
    rank, world_size = setup(
        distributed=distributed
    )
    
    dataset = UnifiedDataset(
        DATASET_CONFIG_PATH
    )
    
    if subset_length != None:
        dataset = Subset(dataset, list(range(subset_length)))
    
    dataloader, sampler = get_dataloader(
        dataset=dataset,
        distributed=distributed,
    )
    
    metric_handler = MetricHandler(
        metrics=list(dataset.metrics.keys()), 
        grid=dataset.grid
    )    
    
    if distributed:
        sampler.set_epoch(0)  # Needed for DistributedSampler

    for i, sample in enumerate(dataloader):
        print(f"Rank {rank} Batch {i}, x.mean() = {sample['x'].mean().item()}")
        output = metric_handler(sample)
        print(f'r{rank} {i} {output.keys()}')

    if distributed:
        dist.destroy_process_group()
    
    
if __name__=="__main__":
    main(
        distributed=False,
        subset_length=10,
    )
import os

import zarr
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from datetime import datetime, timedelta

def idx_to_datetime(
        idx: int, 
        base_time: datetime = datetime(2025, 8, 5, 21, 0, 0), 
        time_delta: timedelta = timedelta(minutes=30),
    ) -> datetime:

    return base_time + idx * time_delta

class ZarrSaver(nn.Module):
    """
    A torch.nn.Module that saves tensors to a Zarr archive. 🗄️

    This module is flexible and handles both batched inputs and
    singleton samples. For each sample, it saves the tensor from
    `sample['x']` into a Zarr archive under a key from `sample['time']`.
    """
    def __init__(self, path: str, mode: str = 'a'):
        """
        Initializes the ZarrSaver module.

        Args:
            path (str): The path to the Zarr archive directory.
            mode (str, optional): The mode to open the Zarr store.
                                  Defaults to 'a' (append/read/write).
        """
        super().__init__()
        self.path = path
        self.root = zarr.open(self.path, mode=mode)
        print(f"Zarr archive opened at '{path}' in mode '{mode}'.")

    def forward(self, sample: dict) -> dict:
        """
        Saves data and passes the sample through, handling both batches and singletons.

        Args:
            sample (dict): A dictionary containing the data.
                           - For a batch:
                             - 'x': Tensor of shape (B, ...).
                             - 'time': List or tuple of B keys.
                           - For a singleton:
                             - 'x': Tensor of shape (...).
                             - 'time': A single key (e.g., string, int).

        Returns:
            dict: The original input sample, unchanged.
        """
        data_tensor = sample['x']
        key_info = sample['time']

        # Check if the input is a batch by seeing if 'time' is a list/tuple
        is_batch = isinstance(key_info, (list, tuple))

        if is_batch:
            # --- BATCH PROCESSING ---
            data_batch = data_tensor.detach().cpu().numpy()
            keys = key_info
            for i in range(data_batch.shape[0]):
                key = str(keys[i])
                data_item = data_batch[i]
                self.root[key] = data_item
        else:
            # --- SINGLETON PROCESSING ---
            key = str(key_info)
            data_item = data_tensor.detach().cpu().numpy()
            self.root[key] = data_item

        # Return the original sample to not disrupt the pipeline
        return sample

# 1. Dummy Dataset (unchanged)
class MyDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Data remains on the CPU
        return {
            'x' : idx * torch.ones(1, 16, 16) ,
            'time': idx_to_datetime(idx),
        }

# 2. Setup the process group for CPU
def setup():
    # These variables are set by SLURM
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])

    # The master address and port are set in the SLURM script
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']

    # ❗ Key change: Use 'gloo' backend for CPU communication
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://{master_addr}:{master_port}",
        # init_method='env://',
        world_size=world_size,
        rank=rank
    )
    print(f'{master_addr} : {master_port}')
    print(f"Process group initialized for rank {rank} of {world_size} on CPU.")


# 3. The main training function
def run():
    setup()

    # Create
    saver = ZarrSaver(path='/p/project1/hclimrep/vozar2/PACE-Toolkit/dirty_bit/test.zarr')

    # Create the dataset
    dataset = MyDataset()

    # Create the DistributedSampler
    sampler = DistributedSampler(dataset)

    # Create the DataLoader
    # Use SLURM_CPUS_PER_TASK to set the number of worker processes
    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,
    )

    # Get the rank for logging
    rank = dist.get_rank()

    # Example of a training loop
    sampler.set_epoch(0)
    for i, batch in enumerate(dataloader):
        # if i == 0 and rank == 0:
        # print(f"Rank {rank}, Batch {i}, Data shape: {batch['x'].mean()}")
        saver(batch)

    dist.destroy_process_group()

if __name__ == "__main__":
    run()
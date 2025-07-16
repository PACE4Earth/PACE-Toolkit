import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

# 1. Dummy Dataset (unchanged)
class MyDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Data remains on the CPU
        return {
            'x' : idx * torch.ones(1, 16, 16) 
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
def train():
    setup()

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
        print(f"Rank {rank}, Batch {i}, Data shape: {batch['x'].mean()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    train()
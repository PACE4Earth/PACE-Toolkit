import os
import sys
import torch
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
global_rank = comm.Get_rank()
size = comm.Get_size()

# Get the hostname to see which node we are on
hostname = os.uname()[1]


try:
    local_rank = int(os.environ['SLURM_LOCALID'])
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')
    
    # Check if CUDA (NVIDIA GPUs) is available
    if not torch.cuda.is_available():
        print(f"Rank {global_rank} on {hostname}: CUDA is not available.", flush=True)
        sys.exit(1)

    logical_gpu_id = torch.cuda.current_device() 
    gpu_name = torch.cuda.get_device_name(logical_gpu_id)
    
    x = torch.ones(512*global_rank, device=logical_gpu_id)
    tensor_mem = x.nelement() * x.element_size() 
    allocated_mem = torch.cuda.memory_allocated()
    
    device = f'cuda:{logical_gpu_id}'
    
    try:
        x = x.to(device)
        print(x.device)
        print(allocated_mem)
    except Exception as e:
        print(e)

except KeyError:
    print(
        f"Rank {global_rank} on {hostname}: Error - 'SLURM_LOCALID' not found.",
        flush=True
    )
    sys.exit(1)

# Barrier to sync all processes before exiting
comm.Barrier()

if global_rank == 0:
    print("\n✅ All MPI ranks have successfully reported their GPU assignments.")
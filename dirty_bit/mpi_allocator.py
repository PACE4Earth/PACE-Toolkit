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
    # --- VERIFICATION VARIABLES ---
    # We still get local_rank, but just for printing and understanding.
    local_rank = int(os.environ['SLURM_LOCALID'])
    # This is the key variable set by Slurm to isolate GPUs.
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')
    
    # Check if CUDA (NVIDIA GPUs) is available
    if not torch.cuda.is_available():
        print(f"Rank {global_rank} on {hostname}: CUDA is not available.", flush=True)
        sys.exit(1)

    # --- NO LONGER NEEDED ---
    # torch.cuda.set_device(local_rank)  <-- REMOVED THIS LINE
    # By default, PyTorch will use the only device it can see (device 0),
    # which Slurm has already mapped to a unique physical GPU.

    # Get info about the GPU this process is using
    # It will always be '0' from the process's point of view.
    logical_gpu_id = torch.cuda.current_device() 
    gpu_name = torch.cuda.get_device_name(logical_gpu_id)

    # Print information from each rank
    print(
        f"Hello from Global Rank {global_rank}/{size} on node '{hostname}'.\n"
        f"  - My SLURM Local ID is: {local_rank}\n"
        f"  - Slurm assigned me physical GPU(s): CUDA_VISIBLE_DEVICES={visible_devices}\n"
        f"  - I am using logical GPU: {logical_gpu_id} ({gpu_name})\n"
        f"-------------------------------------------------",
        flush=True
    )

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
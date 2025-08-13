#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=job_output.%j
#SBATCH --error=job_error.%j
#SBATCH --account=weatherai

# 🚀 Specify the Booster partition and request GPU resources
#SBATCH --partition=booster
#SBATCH --gres=gpu:4

# Purge existing modules to ensure a clean environment
module --force purge

# 📦 Load the module stack for the GPU environment on JUWELS Booster
# This stack includes CUDA and CUDA-aware versions of MPI and PyTorch.
ml Stages/2024 GCCcore/.12.3.0 GCC/12.3.0 zarr/2.18.3 CUDA/12
ml SciPy-Stack/2023a PyTorch/2.1.2 netcdf4-python/1.6.4-serial
ml ParaStationMPI/5.9.2-1 mpi4py/3.1.4

# Set path to GraphCast and ERA5 data (no changes needed)
export GRAPHCAST_EXTENDED_DATA_PATH="/p/scratch/weatherai/shared/ERA5_data_graphcast_inference/240h/2021"
export ERA5_DATA_PATH="/p/scratch/weatherai/shared/ERA5_data_download/2021"
export OUTPUTS_DIR_PATH="/p/scratch/hclimrep/vozar2/outputs"

# Set the master address and port for distributed training (no changes needed)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# Set network interface for the GLOO backend (InfiniBand is correct)
# Note: For multi-node GPU training, the 'nccl' backend is often preferred 
# and typically finds the correct interface automatically.
export GLOO_SOCKET_IFNAME=ib0

# Run the Python script with srun (no changes needed)
srun python evaluator.py
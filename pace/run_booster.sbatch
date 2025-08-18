#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=logs/booster.%j.out
#SBATCH --error=logs/booster.%j.err
#SBATCH --account=weatherai

# 🚀 Specify the Booster partition and request GPU resources
#SBATCH --partition=booster
#_#SBATCH --gres=gpu:4

# Purge existing modules to ensure a clean environment
module --force purge

# 📦 Load the module stack for the GPU environment on JUWELS Booster
# This stack includes CUDA and CUDA-aware versions of MPI and PyTorch.
ml Stages/2024 GCCcore/.12.3.0 GCC/12.3.0 zarr/2.18.3 CUDA/12
ml SciPy-Stack/2023a PyTorch/2.1.2 netcdf4-python/1.6.4-serial
ml ParaStationMPI/5.9.2-1 mpi4py/3.1.4

echo "#########################################################"
echo "###           STARTING BOOSTER JOB SCRIPT             ###"
echo "#########################################################"

# [SUGGESTED CHANGE 1] Set NCCL_DEBUG to INFO for verbose collective communication logging.
# This will show details about how GPUs are communicating with each other.
export NCCL_DEBUG=INFO

# Set the master address and port for distributed training (no changes needed)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# [SUGGESTED CHANGE 2] Print the top-level CUDA_VISIBLE_DEVICES allocated to the entire job.
# SLURM sets this for the main job script.
echo "SLURM allocated devices for the job: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "---------------------------------------------------------"

# export GLOO_SOCKET_IFNAME=ib0

# Run the Python script with srun (no changes needed)
srun --verbose python evaluator.py
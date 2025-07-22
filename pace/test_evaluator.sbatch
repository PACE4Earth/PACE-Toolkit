#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00
#SBATCH --output=job_output.%j
#SBATCH --error=job_error.%j
#SBATCH --account=weatherai

# ❗ Load modules for a CPU environment
ml Stages/2025 GCC/13.3.0 GCCcore/.13.3.0 SciPy-Stack/2024a PyTorch/2.5.1 netcdf4-python/1.7.1.post2-serial
ml ParaStationMPI/5.11.0-1

# Set path to GraphCast data
export GRAPHCAST_EXTENDED_DATA_PATH="/p/scratch/weatherai/shared/ERA5_data_graphcast_inference/240h/2021"

# Set the master address and port
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# ?
export GLOO_SOCKET_IFNAME=ib0

# Run the training script
srun python evaluator.py
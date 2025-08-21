#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --ntasks=8
#SBATCH --partition=booster
#SBATCH --time=00:15:00
#SBATCH --output=logs/booster.%j.out
#SBATCH --error=logs/booster.%j.err
#SBATCH --account=weatherai


module --force purge
ml Stages/2024 GCCcore/.12.3.0 GCC/12.3.0 zarr/2.18.3 CUDA/12
ml SciPy-Stack/2023a PyTorch/2.1.2 netcdf4-python/1.6.4-serial
ml ParaStationMPI/5.9.2-1 mpi4py/3.1.4

export OUTPUT_DIR_PATH="/p/scratch/hclimrep/${USER}/outputs_booster"
export PLOTS_DIR_PATH="/p/scratch/hclimrep/${USER}/outputs_booster/plots"

mkdir -p "$PLOTS_DIR_PATH"

export NCCL_DEBUG=INFO

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

srun --verbose python evaluator.py

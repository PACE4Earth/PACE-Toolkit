#!/bin/bash
#SBATCH --job-name=multinode_gpu_test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=8
#SBATCH --gpus-per-node=4
#SBATCH --partition=booster
#SBATCH --time=00:05:00
#SBATCH --output=mpi_alloc.%j.out
#SBATCH --error=mpi_alloc.%j.err
#SBATCH --account=weatherai

module --force purge

ml Stages/2024 GCCcore/.12.3.0 GCC/12.3.0 zarr/2.18.3 CUDA/12
ml SciPy-Stack/2023a PyTorch/2.1.2 netcdf4-python/1.6.4-serial
ml ParaStationMPI/5.9.2-1 mpi4py/3.1.4

srun python3 mpi_allocator.py

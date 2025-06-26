#!/bin/bash
#SBATCH --account=hclimrep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --output=mpi-out.%j
#SBATCH --error=mpi-err.%j
#SBATCH --time=00:05:00

module load Stages/2025 GCCcore/.13.3.0 SciPy-Stack/2024a PyTorch/2.5.1 netcdf4-python/1.7.1.post2-serial

srun python ./pace/evaluator.py
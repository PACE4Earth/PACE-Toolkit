#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:05:00
#SBATCH --output=job_output.%j
#SBATCH --error=job_error.%j
#SBATCH --account=weatherai

# ❗ Load modules for a CPU environment 
module --force purge
ml Stages/2024 GCCcore/.12.3.0 GCC/12.3.0 zarr/2.18.3
ml SciPy-Stack/2023a PyTorch/2.1.2 netcdf4-python/1.6.4-serial
ml ParaStationMPI/5.9.2-1 mpi4py/3.1.4

# Set the master address and port
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# ?
export GLOO_SOCKET_IFNAME=ib0

# Run the training script
srun python ./mpi_saver.py
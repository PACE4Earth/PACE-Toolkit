#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:15:00
#SBATCH --output=cluster.%j.out
#SBATCH --error=cluster.%j.err
#SBATCH --account=weatherai

# backup modules, in case something goes wrong
# ml Stages/2025 GCC/13.3.0 GCCcore/.13.3.0 SciPy-Stack/2024a PyTorch/2.5.1 netcdf4-python/1.7.1.post2-serial
# ml ParaStationMPI/5.11.0-1

module --force purge
ml Stages/2024 GCCcore/.12.3.0 GCC/12.3.0 zarr/2.18.3
ml SciPy-Stack/2023a PyTorch/2.1.2 netcdf4-python/1.6.4-serial
ml ParaStationMPI/5.9.2-1 mpi4py/3.1.4

# Set path to GraphCast data
export GRAPHCAST_EXTENDED_DATA_PATH="/p/scratch/weatherai/shared/ERA5_data_graphcast_inference/240h/2021"
export ERA5_DATA_PATH="/p/scratch/weatherai/shared/ERA5_data_download/2021"
# export OUTPUTS_DIR_PATH="/p/scratch/hclimrep/shared/pace/outputs"
export OUTPUTS_DIR_PATH="/p/scratch/hclimrep/shared/pace/vozar2"

# Set the master address and port
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# ?
export GLOO_SOCKET_IFNAME=ib0

# Run the training script
srun python evaluator.py
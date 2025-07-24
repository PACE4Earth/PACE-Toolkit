import os
import json
from pathlib import Path
import xarray as xr
import numpy as np
from collections import defaultdict
import datetime
import torch
import torch.distributed as dist
from torch.utils.data import (
    Subset,
    DataLoader,
    DistributedSampler,
    RandomSampler,
)

from utils.dataset import UnifiedDataset
from metrics.metric_handler import MetricHandler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CONFIG_PATH =  os.path.join(BASE_DIR, 'configs', 'graphcast_extended.json')

def setup(distributed=False):
    if distributed:
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']

        dist.init_process_group(
            backend="gloo",
            init_method=f"tcp://{master_addr}:{master_port}",
            world_size=world_size,
            rank=rank
        )
        print('__________________________________________________')
        print(f'{master_addr} : {master_port}')
        print(f"Process group initialized for rank {rank} of {world_size} on CPU.")
        print('__________________________________________________')
    else:
        rank = 0
        world_size = 1

    return rank, world_size

def get_dataloader(dataset, distributed=False):
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)

    num_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', 0))
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=num_workers,
        shuffle=False,
    )
    return dataloader, sampler

def main(distributed=False, subset_length=None):
    rank, world_size = setup(distributed=distributed)

    # Load dataset config
    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Load the full dataset (UnifiedDataset instance)
    full_dataset = UnifiedDataset(DATASET_CONFIG_PATH)

    # Use a subset if requested, but keep the full_dataset reference
    if subset_length is not None:
        dataset = Subset(full_dataset, list(range(subset_length)))
    else:
        dataset = full_dataset

    # Prepare dataloader
    dataloader, sampler = get_dataloader(dataset=dataset, distributed=distributed)
    print('Dataset lenght:\t', dataset.__len__())

    # Metric handler setup
    metric_handler = MetricHandler(
        metrics=list(full_dataset.metrics.keys()),
        grid=full_dataset.grid
    )

    # If using distributed evaluation
    if distributed:
        sampler.set_epoch(0)

    # Evaluation loop
    with torch.no_grad():
        base_time_data = defaultdict(lambda: defaultdict(list))  # base_time -> var -> [leadtime slices]
        base_time_coords = {}  # Store coords separately per base_time
        collected_leadtimes = defaultdict(list)  # base_time -> [leadtime_hours]

        for i, sample in enumerate(dataloader):
            valid_time = sample['lead_time'].item()
            base_time = sample['base_time'].item()
            leadtime_hours = int((valid_time - base_time) / 3600)

            # Convert base_time to datetime for filename
            base_dt = datetime.datetime.utcfromtimestamp(base_time)

            output = metric_handler(sample)

            # clean up with log_output handler, abstraction
            if base_time not in base_time_coords:
                base_time_coords[base_time] = {
                    "lat": full_dataset.grid["lat"].numpy(),
                    "lon": full_dataset.grid["lon"].numpy(),
                    "level": [
                        1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225,
                        250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800,
                        825, 850, 875, 900, 925, 950, 975, 1000
                    ]
                }

            for key, val in output.items():
                
                if key != 'correlation':
                    if isinstance(val, torch.Tensor):
                        val_np = val.cpu().numpy()

                        # Remove batch/time dimension if singleton
                        if val_np.ndim == 4 and val_np.shape[0] == 1:
                            val_np = val_np.squeeze(0)

                        base_time_data[base_time][key].append((leadtime_hours, val_np))

            # Optional: add geopotential from sample
            if 'geopotential' in sample:
                geo = sample['geopotential'].cpu().numpy()
                if geo.ndim == 4 and geo.shape[0] == 1:
                    geo = geo.squeeze(0)
                base_time_data[base_time]['geopotential'].append((leadtime_hours, geo))

            collected_leadtimes[base_time].append(leadtime_hours)

        if 'correlation' in metric_handler.metrics.keys():
            
            fig, ax = metric_handler.metrics['correlation'].evaluate_corr()
            
            fig.savefig(
                    os.path.join(BASE_DIR, 'plots', 'corrs.png')
                )    
            
            for key in ['t2m_u10m', 't2m_v10m', 't2m_mslp', 'u10m_v10m', 'u10m_mslp', 'v10m_mslp']:
                fig, ax = metric_handler.metrics['correlation'].visualize(key)
                fig.savefig(
                    os.path.join(BASE_DIR, 'plots', f'bivariate_{key}.png')
                )                

        # Now save one file per base_time
        for base_time, var_data in base_time_data.items():
            base_dt = datetime.datetime.utcfromtimestamp(base_time)
            leadtimes_sorted = sorted(set(collected_leadtimes[base_time]))

            coords = base_time_coords[base_time]
            coords["lead_time"] = leadtimes_sorted
            coords["base_time"] = base_dt  # scalar

            data_vars = {}
            for var_name, values in var_data.items():
                # Sort by lead_time
                values_sorted = sorted(values, key=lambda x: x[0])  # sort by leadtime
                lead_vals = [v for _, v in values_sorted]

                arr = np.stack(lead_vals, axis=0)  # Shape: (lead_time, ...)
                if arr.ndim == 4:
                    dims = ("lead_time", "level", "lat", "lon")
                elif arr.ndim == 3:
                    dims = ("lead_time", "lat", "lon")
                elif arr.ndim == 1:
                    dims = ("value")
                else:
                    raise ValueError(f"Unsupported output shape {arr.shape} for variable {var_name}")

                data_vars[var_name] = (dims, arr)

            ds_out = xr.Dataset(
                data_vars=data_vars,
                coords=coords,
            )

            save_dir = os.path.join(BASE_DIR, "outputs", full_dataset.name)
            os.makedirs(save_dir, exist_ok=True)

            out_path = os.path.join(save_dir, f"{base_dt.strftime('%Y%m%d_%H')}.nc")
            ds_out.to_netcdf(out_path)
            print(f"Saved to {out_path}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main(
        distributed=False,
        subset_length=40,
    )

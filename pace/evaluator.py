import os
import json
import random
import shutil
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
DATASET_CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'dataset_config.json')

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
        print(f"Process group initialized for rank {rank} of {world_size} on CPU.")
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

def save_dataset_to_netcdf(dataset_dict, coords_dict, save_dir, job_name, leadtime_dict):
    os.makedirs(save_dir, exist_ok=True)
    for base_time, var_data in dataset_dict.items():
        base_dt = datetime.datetime.utcfromtimestamp(base_time)
        leadtimes_sorted = sorted(set(leadtime_dict[base_time]))

        coords = {
            "level": coords_dict[base_time]["level"],
            "lead_time": leadtimes_sorted,
            "base_time": base_dt,
        }

        data_vars = {}
        for var_name, values in var_data.items():
            # Sort by lead time
            values_sorted = sorted(values, key=lambda x: x[0])
            lead_vals = [v for _, v in values_sorted]
            arr = np.stack(lead_vals, axis=0)

            if arr.ndim == 4:
                # spatial dims mean
                arr = arr.mean(axis=(-1, -2))
                dims = ("lead_time", "level")
            elif arr.ndim == 3:
                arr = arr.mean(axis=-1)
                dims = ("lead_time", "level")
            elif arr.ndim == 2:
                dims = ("lead_time", "level")
            elif arr.ndim == 1:
                dims = ("lead_time",)
            else:
                raise ValueError(f"Unsupported shape for summary var '{var_name}': {arr.shape}")

            data_vars[var_name] = (dims, arr)

        ds_out = xr.Dataset(data_vars=data_vars, coords=coords)
        out_path = os.path.join(save_dir, f"{job_name}_{base_dt.strftime('%Y%m%d_%H')}.nc")
        ds_out.to_netcdf(out_path)
        print(f"Saved summary to {out_path}")

def main(distributed=False):
    rank, world_size = setup(distributed=distributed)

    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    model_dataset = UnifiedDataset(DATASET_CONFIG_PATH, dataset_key="model")
    reference_dataset = UnifiedDataset(DATASET_CONFIG_PATH, dataset_key="reference") if "reference" in config.get("datasets", {}) else None
    model_name = config["datasets"]["model"]["name"]
    reference_name = config["datasets"].get("reference", {}).get("name")

    # Select matching fullfield samples and update flags in datasets
    fullfield_sample_indices = model_dataset.select_matching_fullfield_samples(reference_dataset or model_dataset)

    metric_handler = MetricHandler(
        metrics=list(model_dataset.metrics.keys()),
        grid=model_dataset.grid
    )

    model_outputs = defaultdict(lambda: defaultdict(list))
    reference_outputs = defaultdict(lambda: defaultdict(list))
    model_coords = defaultdict(dict)
    reference_coords = defaultdict(dict)
    leadtimes_model = defaultdict(list)
    leadtimes_ref = defaultdict(list)

    # --- MODEL SUMMARY ---
    dataset = model_dataset
    dataloader, sampler = get_dataloader(dataset, distributed=distributed)
    if distributed:
        sampler.set_epoch(0)

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            base_dt = sample["base_time"]
            lead_dt = sample["lead_time"]
            leadtime_hours = int(lead_dt.total_seconds() / 3600)
            base_time = int(base_dt.timestamp())

            output = metric_handler(sample)

            if base_time not in model_coords:
                model_coords[base_time] = {
                    "level": model_dataset.grid["pressure_levels"].numpy()
                }

            for key, val in output.items():
                if isinstance(val, torch.Tensor):
                    val_np = val.squeeze(0).cpu().numpy() if val.ndim == 4 and val.shape[0] == 1 else val.cpu().numpy()
                    model_outputs[base_time][key].append((leadtime_hours, val_np))

            leadtimes_model[base_time].append(leadtime_hours)

    # --- REFERENCE SUMMARY ---
    if reference_dataset:
        reference_dataloader, _ = get_dataloader(reference_dataset, distributed=distributed)
        if distributed:
            # Optional: set epoch if using distributed sampler
            pass
        with torch.no_grad():
            for i, sample in enumerate(reference_dataloader):
                base_dt = sample["base_time"]
                lead_dt = sample["lead_time"]
                leadtime_hours = int(lead_dt.total_seconds() / 3600)
                base_time = int(base_dt.timestamp())

                output = metric_handler(sample)

                if base_time not in reference_coords:
                    reference_coords[base_time] = {
                        "level": reference_dataset.grid["pressure_levels"].numpy()
                    }

                for key, val in output.items():
                    if isinstance(val, torch.Tensor):
                        val_np = val.squeeze(0).cpu().numpy() if val.ndim == 4 and val.shape[0] == 1 else val.cpu().numpy()
                        reference_outputs[base_time][key].append((leadtime_hours, val_np))

                leadtimes_ref[base_time].append(leadtime_hours)

    summary_dir = os.path.join(BASE_DIR, "outputs", "summary", model_name)
    save_dataset_to_netcdf(model_outputs, model_coords, summary_dir, job_name=f"job{rank}", leadtime_dict=leadtimes_model)

    if reference_dataset:
        summary_dir_ref = os.path.join(BASE_DIR, "outputs", "summary", reference_name)
        save_dataset_to_netcdf(reference_outputs, reference_coords, summary_dir_ref, job_name=f"job{rank}", leadtime_dict=leadtimes_ref)

    # --- Save fullfield outputs ---
    for tag in [model_name, reference_name]:
        if tag is None:
            continue
        full_dir = os.path.join(BASE_DIR, "outputs", "fullfields", tag)
        if os.path.exists(full_dir):
            shutil.rmtree(full_dir)
        os.makedirs(full_dir)

    # Iterate only over dataset indices where fullfield_sample_flags is True
    # The select_matching_fullfield_samples updates these flags accordingly
    for dataset, tag in zip([model_dataset, reference_dataset], [model_name, reference_name]):
        if dataset is None or tag is None:
            continue

        for idx, flag in enumerate(dataset.fullfield_sample_flags):
            if not flag:
                continue

            sample = dataset[idx]
            metrics = metric_handler(sample)
            base_dt = sample["base_time"]
            lead_dt = sample["lead_time"]
            leadtime_hours = int(lead_dt.total_seconds() / 3600)

            data_vars = {}
            coords = {
                "lat": dataset.grid["lat"].numpy(),
                "lon": dataset.grid["lon"].numpy(),
                "level": dataset.grid["pressure_levels"].numpy(),
                "base_time": base_dt,
                "lead_time": leadtime_hours
            }

            for key, val in metrics.items():
                if isinstance(val, torch.Tensor):
                    arr = val.squeeze(0).cpu().numpy() if val.ndim == 4 and val.shape[0] == 1 else val.cpu().numpy()
                    if arr.ndim == 3:
                        dims = ("level", "lat", "lon")
                        data_vars[key] = (dims, arr)

            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            out_path = os.path.join(BASE_DIR, "outputs", "fullfields", tag, f"valid{base_dt.strftime('%Y%m%d_%H')}_lead{leadtime_hours:03d}.nc")
            ds.to_netcdf(out_path)
            print(f"Saved full field to {out_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(distributed=False)

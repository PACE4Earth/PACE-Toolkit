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
            values_sorted = sorted(values, key=lambda x: x[0])
            lead_vals = [v for _, v in values_sorted]
            arr = np.stack(lead_vals, axis=0)

            if arr.ndim == 4:
                arr = arr.mean(axis=(-1, -2))  # spatial mean -> (lead, level)
                dims = ("lead_time", "level")
            elif arr.ndim == 3:
                arr = arr.mean(axis=-1)  # -> (lead, level) or similar
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

def main(distributed=False, subset_length=None):
    rank, world_size = setup(distributed=distributed)

    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    model_dataset = UnifiedDataset(DATASET_CONFIG_PATH, dataset_key="model")
    reference_dataset = UnifiedDataset(DATASET_CONFIG_PATH, dataset_key="reference") if "reference" in config.get("datasets", {}) else None
    model_name = config["datasets"]["model"]["name"]
    reference_name = config["datasets"]["reference"]["name"] if reference_dataset else None
    n_fullfield_samples = config.get("visualizations", {}).get("n_fullfield_samples", 10)

    dataset = Subset(model_dataset, list(range(subset_length))) if subset_length is not None else model_dataset
    dataloader, sampler = get_dataloader(dataset, distributed=distributed)

    if distributed:
        sampler.set_epoch(0)

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
    stored_samples = []

    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            base_time = sample["base_time"].item()
            valid_time = sample["lead_time"].item()
            leadtime_hours = int((valid_time - base_time) / 3600)

            base_dt = datetime.datetime.utcfromtimestamp(base_time)
            valid_dt = datetime.datetime.utcfromtimestamp(valid_time)

            output = metric_handler(sample)
            ref_output = None
            if reference_dataset:
                reference_sample = reference_dataset[i]
                ref_output = metric_handler(reference_sample)

            if base_time not in model_coords:
                model_coords[base_time] = {
                    "level": model_dataset.grid["pressure_levels"].numpy()
                }
            if base_time not in reference_coords and ref_output is not None:
                reference_coords[base_time] = {
                    "level": reference_dataset.grid["pressure_levels"].numpy()
                }

            for key, val in output.items():
                if isinstance(val, torch.Tensor):
                    val_np = val.squeeze(0).cpu().numpy() if val.ndim == 4 and val.shape[0] == 1 else val.cpu().numpy()
                    model_outputs[base_time][key].append((leadtime_hours, val_np))

            if ref_output is not None:
                for key, val in ref_output.items():
                    if isinstance(val, torch.Tensor):
                        val_np = val.squeeze(0).cpu().numpy() if val.ndim == 4 and val.shape[0] == 1 else val.cpu().numpy()
                        reference_outputs[base_time][key].append((leadtime_hours, val_np))

            leadtimes_model[base_time].append(leadtime_hours)
            if ref_output is not None:
                leadtimes_ref[base_time].append(leadtime_hours)

            total_seen = len(stored_samples)
            if len(stored_samples) < n_fullfield_samples:
                stored_samples.append((output, ref_output, base_dt, valid_dt, leadtime_hours))
            else:
                replace_idx = random.randint(0, total_seen)
                if replace_idx < n_fullfield_samples:
                    stored_samples[replace_idx] = (output, ref_output, base_dt, valid_dt, leadtime_hours)

    summary_dir = os.path.join(BASE_DIR, "outputs", "summary", model_name)
    save_dataset_to_netcdf(model_outputs, model_coords, summary_dir, job_name=f"job{rank}", leadtime_dict=leadtimes_model)

    if reference_dataset:
        summary_dir_ref = os.path.join(BASE_DIR, "outputs", "summary", reference_name)
        save_dataset_to_netcdf(reference_outputs, reference_coords, summary_dir_ref, job_name=f"job{rank}", leadtime_dict=leadtimes_ref)

    for tag in [model_name, reference_name]:
        full_dir = os.path.join(BASE_DIR, "outputs", "fullfields", tag)
        if os.path.exists(full_dir):
            shutil.rmtree(full_dir)
        os.makedirs(full_dir)

    for idx, (model_metrics, ref_metrics, base_dt, valid_dt, leadtime_hours) in enumerate(stored_samples):
        for tag, metrics, grid in zip(
            [model_name, reference_name],
            [model_metrics, ref_metrics],
            [model_dataset.grid, reference_dataset.grid if reference_dataset else None]
        ):
            if metrics is None:
                continue

            data_vars = {}
            coords = {
                "lat": grid["lat"].numpy(),
                "lon": grid["lon"].numpy(),
                "level": grid["pressure_levels"].numpy(),
                "base_time": base_dt,
                "valid_time": valid_dt,
                "lead_time": leadtime_hours
            }

            for key, val in metrics.items():
                if isinstance(val, torch.Tensor):
                    arr = val.squeeze(0).cpu().numpy() if val.ndim == 4 and val.shape[0] == 1 else val.cpu().numpy()
                    if arr.ndim == 3:
                        dims = ("level", "lat", "lon")
                        data_vars[key] = (dims, arr)

            ds = xr.Dataset(data_vars=data_vars, coords=coords)
            out_path = os.path.join(BASE_DIR, "outputs", "fullfields", tag, f"valid{valid_dt.strftime('%Y%m%d_%H')}.nc")
            ds.to_netcdf(out_path)
            print(f"Saved full field to {out_path}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main(distributed=False, subset_length=40)

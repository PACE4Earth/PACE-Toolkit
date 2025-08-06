import os
import json
import xarray as xr
import numpy as np
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

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


def save_fullfields_to_netcdf(output_dict, coords_dict, save_path):
    """
    Save aggregated outputs to a single NetCDF file with shape:
    (base_time, lead_time, level, lat, lon)
    """
    ds_out = xr.Dataset(
        {k: (("base_time", "lead_time", "level", "lat", "lon"), v) for k, v in output_dict.items()},
        coords=coords_dict
    )
    ds_out.to_netcdf(save_path)
    print(f"Saved aggregated fullfield file: {save_path}")


def aggregate_samples(metric_data, base_times, lead_times, levels, lats, lons):
    """
    Aggregate samples into shape (n_base, n_lead, n_level, n_lat, n_lon)
    """
    unique_base_times = sorted(set(base_times))
    unique_lead_times = sorted(set(lead_times))

    n_base = len(unique_base_times)
    n_lead = len(unique_lead_times)
    n_level = len(levels)
    n_lat = len(lats)
    n_lon = len(lons)

    aggregated = {k: np.full((n_base, n_lead, n_level, n_lat, n_lon), np.nan, dtype=np.float32)
                  for k in metric_data.keys()}

    for idx, (bt, lt) in enumerate(zip(base_times, lead_times)):
        i_base = unique_base_times.index(bt)
        i_lead = unique_lead_times.index(lt)
        for key in metric_data:
            aggregated[key][i_base, i_lead] = metric_data[key][idx]

    coords = {
        "base_time": np.array(unique_base_times, dtype="datetime64[ns]"),
        "lead_time": np.array(unique_lead_times, dtype="timedelta64[ns]"),
        "level": levels,
        "lat": lats,
        "lon": lons
    }
    return aggregated, coords


def main(distributed=False):
    rank, world_size = setup(distributed=distributed)

    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Expand outputs_dir from config or use default
    outputs_dir = os.path.expandvars(config.get("outputs_dir", os.path.join(BASE_DIR, "outputs")))
    os.makedirs(outputs_dir, exist_ok=True)

    model_dataset = UnifiedDataset(DATASET_CONFIG_PATH, dataset_key="model")
    reference_dataset = UnifiedDataset(
        DATASET_CONFIG_PATH,
        dataset_key="reference",
        shared_valid_times=model_dataset.chosen_valid_times
    ) if "reference" in config.get("datasets", {}) else None

    model_name = config["datasets"]["model"]["name"]
    reference_name = config["datasets"].get("reference", {}).get("name")

    metric_handler = MetricHandler(
        metrics=list(model_dataset.metrics.keys()),
        grid=model_dataset.grid
    )

    # --- MODEL FULLFIELDS ---
    model_metrics_data = defaultdict(list)
    base_times_model = []
    lead_times_model = []

    dataloader, sampler = get_dataloader(model_dataset, distributed=distributed)
    if distributed:
        sampler.set_epoch(0)

    with torch.no_grad():
        for sample in dataloader:
            metrics = metric_handler(sample)
            base_dt = sample["base_time"]
            lead_dt = sample["lead_time"]

            base_times_model.append(base_dt)
            lead_times_model.append(lead_dt)

            for key, val in metrics.items():
                if isinstance(val, torch.Tensor):
                    arr = val.squeeze(0).cpu().numpy()  # (level, lat, lon)
                    model_metrics_data[key].append(arr)

    for key in model_metrics_data:
        model_metrics_data[key] = np.stack(model_metrics_data[key], axis=0)

    aggregated_model, model_coords = aggregate_samples(
        model_metrics_data,
        base_times_model,
        lead_times_model,
        levels=model_dataset.grid["pressure_levels"].numpy(),
        lats=model_dataset.grid["lat"].numpy(),
        lons=model_dataset.grid["lon"].numpy()
    )

    save_fullfields_to_netcdf(aggregated_model, model_coords, os.path.join(outputs_dir, f"{model_name}_fullfields.nc"))

    # --- REFERENCE FULLFIELDS ---
    if reference_dataset:
        reference_metrics_data = defaultdict(list)
        base_times_ref = []
        lead_times_ref = []

        ref_dataloader, _ = get_dataloader(reference_dataset, distributed=distributed)

        with torch.no_grad():
            for sample in ref_dataloader:
                metrics = metric_handler(sample)
                base_dt = sample["base_time"]
                lead_dt = sample["lead_time"]

                base_times_ref.append(base_dt)
                lead_times_ref.append(lead_dt)

                for key, val in metrics.items():
                    if isinstance(val, torch.Tensor):
                        arr = val.squeeze(0).cpu().numpy()
                        reference_metrics_data[key].append(arr)

        for key in reference_metrics_data:
            reference_metrics_data[key] = np.stack(reference_metrics_data[key], axis=0)

        aggregated_ref, reference_coords = aggregate_samples(
            reference_metrics_data,
            base_times_ref,
            lead_times_ref,
            levels=reference_dataset.grid["pressure_levels"].numpy(),
            lats=reference_dataset.grid["lat"].numpy(),
            lons=reference_dataset.grid["lon"].numpy()
        )

        save_fullfields_to_netcdf(aggregated_ref, reference_coords, os.path.join(outputs_dir, f"{reference_name}_fullfields.nc"))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(distributed=False)

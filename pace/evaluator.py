import os
import json
import xarray as xr
import numpy as np
from collections import defaultdict
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from utils.dataset import UnifiedDataset
from utils.output_logger import IndexedZarrSaver
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

def main(distributed=False):
    rank, world_size = setup(distributed=distributed)

    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

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

    model_output_logger = IndexedZarrSaver(path=os.path.join(outputs_dir, f"{model_name}.zarr"))
    reference_output_logger = IndexedZarrSaver(path=os.path.join(outputs_dir, f"{reference_name}.zarr")) if reference_dataset else None

    metric_handler = MetricHandler(
        metrics=list(model_dataset.metrics.keys()),
        grid=model_dataset.grid
    )

    def evaluate_and_log(dataset, logger, dataset_name):
        dataloader, sampler = get_dataloader(dataset, distributed=distributed)
        if distributed:
            sampler.set_epoch(0)

        with torch.no_grad():
            for sample in dataloader:
                metrics = metric_handler(sample)
                sample_out = {**metrics, "base_time": sample["base_time"], "lead_time": sample["lead_time"]}
                logger(sample_out)

    # Evaluate model
    evaluate_and_log(model_dataset, model_output_logger, dataset_name=model_name)

    # Save reference outputs if available
    if reference_dataset:
        def passthrough_logger(sample):
            sample_out = {
                name: tensor for name, tensor in sample.items()
                if name not in ["lat", "lon"] and torch.is_tensor(tensor)
            }
            sample_out["base_time"] = sample["base_time"]
            sample_out["lead_time"] = sample["lead_time"]
            reference_output_logger(sample_out)

        evaluate_and_log(reference_dataset, passthrough_logger, dataset_name=reference_name)

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main(distributed=False)

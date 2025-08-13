import os
import json
from mpi4py import MPI
from collections import defaultdict
import time
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

import xarray.backends.zarr
from xarray.core.utils import is_dict_like

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from utils.dataset import UnifiedDataset
from utils.output_logger import MPIZarrSaver, ZarrDataset
from metrics.metric_handler import MetricHandler
from utils.functions import (
    setup,
    get_dataloader,
    build_dataset_info,
    harmonize_zarr_to_xarray,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CONFIG_PATH = os.path.join(BASE_DIR, 'configs', 'config.json')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.environ['DEVICE'] = DEVICE

def main(distributed=False):
        
    time_start = time.perf_counter()
    rank, world_size = setup(distributed=distributed)
    comm = MPI.COMM_WORLD
    assert world_size == comm.Get_size()
    assert rank == comm.Get_rank()

    if comm.Get_rank() == 0:
        ...
    comm.Barrier()
    
    with open(DATASET_CONFIG_PATH, 'r') as f:
        config = json.load(f)

    outputs_dir = Path(os.path.expandvars(config.get("outputs_dir", "")))
    if not outputs_dir.exists():
        outputs_dir = Path(__file__).resolve().parent / "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    # print('output dir:', outputs_dir)

    # RANK 0 builds the full dataset and sample list
    if rank == 0:
        model_info = build_dataset_info(DATASET_CONFIG_PATH, dataset_key="model")
    else:
        model_info = None

    model_info = comm.bcast(model_info, root=0)
    rank_samples = model_info["samples"][rank::world_size]

    model_dataset = UnifiedDataset.from_sample_list(
        sample_list=rank_samples,
        grid=model_info["grid"],
        metrics=model_info["metrics"],
        requested_names=model_info["requested_names"],
        canonical_names=model_info["canonical_names"],
        config_path=DATASET_CONFIG_PATH,
        dataset_key="model"
    )

    # Repeat the same for reference dataset, if present
    if "reference" in config.get("datasets", {}):
        if rank == 0:
            ref_info = build_dataset_info(
                DATASET_CONFIG_PATH, dataset_key="reference",
                shared_valid_times=model_info["chosen_valid_times"]
            )
        else:
            ref_info = None

        ref_info = comm.bcast(ref_info, root=0)
        ref_rank_samples = ref_info["samples"][rank::world_size]
        reference_dataset = UnifiedDataset.from_sample_list(
            sample_list=ref_rank_samples,
            grid=ref_info["grid"],
            metrics=ref_info["metrics"],
            requested_names=ref_info["requested_names"],
            canonical_names=ref_info["canonical_names"],
            config_path=DATASET_CONFIG_PATH,
            dataset_key="reference"
        )
    else:
        reference_dataset = None

    model_name = config["datasets"]["model"]["name"]
    reference_name = config["datasets"].get("reference", {}).get("name")

    model_output_logger = MPIZarrSaver(
        path=os.path.join(outputs_dir, f"{model_name}.zarr"), 
        comm=comm,
        lat=model_dataset.grid['lat'],
        lon=model_dataset.grid['lon'],
    )

    # Save static coordinates once (only rank 0)
    # if rank == 0:
    #     coords_to_save = {}
    #     for coord_name in ["lat", "lon", "pressure_levels"]:
    #         if coord_name in model_info["grid"]:
    #             coords_to_save[coord_name] = np.array(model_info["grid"][coord_name])
    #     # Save to Zarr root group
    #     zarr_path = os.path.join(outputs_dir, f"{model_name}.zarr")
    #     root = zarr.open(zarr_path, mode="a")
    #     for k, v in coords_to_save.items():
    #         if k not in root:
    #             root.create_dataset(k, data=v, overwrite=True)
    if reference_dataset:
        reference_output_logger = MPIZarrSaver(
            path=os.path.join(outputs_dir, f"{reference_name}.zarr"),
            comm=comm,
            lat=model_dataset.grid['lat'],
            lon=model_dataset.grid['lon'],
        )

        # if rank == 0:
        #     coords_to_save = {}
        #     for coord_name in ["lat", "lon", "pressure_levels"]:
        #         if coord_name in ref_info["grid"]:
        #             coords_to_save[coord_name] = np.array(ref_info["grid"][coord_name])
        #     zarr_path = os.path.join(outputs_dir, f"{reference_name}.zarr")
        #     root = zarr.open(zarr_path, mode="a")
        #     for k, v in coords_to_save.items():
        #         if k not in root:
        #             root.create_dataset(k, data=v, overwrite=True)

    metric_handler = MetricHandler(
        metrics=list(model_dataset.metrics.keys()),
        grid=model_dataset.grid
    )

    def evaluate_and_log(dataset, logger, dataset_name):
        
        if rank==0:
            metrics = metric_handler(dataset[0])
            sample_out = {**metrics, "base_time": dataset[0]["base_time"], "lead_time": dataset[0]["lead_time"]}
            logger.initialize_store(sample_out)
        comm.Barrier()
        
        dataloader, sampler = get_dataloader(dataset, distributed=distributed)
        count = 0
        with torch.no_grad():
            for sample in dataloader:
                metrics = metric_handler(sample)
                sample_out = {**metrics, "base_time": sample["base_time"], "lead_time": sample["lead_time"]}
                logger.save(sample_out)
                count += 1
        print(f"Rank {rank} processed {count} samples.")

    comm.Barrier()

    evaluate_and_log(model_dataset, model_output_logger, dataset_name=model_name)

    print(f"Rank {comm.Get_rank()} waiting at barrier.")
    comm.Barrier()
    print(f"Rank {comm.Get_rank()} passed barrier.")
    
    comm.Barrier()
    if rank==0:
        print(f'Passed barrier after {model_name}.')    
    time.sleep(0.1)

    if reference_dataset:
        evaluate_and_log(reference_dataset, reference_output_logger, dataset_name=reference_name)

    comm.Barrier()
    if rank==0:
        print(f'Passed barrier after {reference_dataset}.')    
    time.sleep(0.1)


    if comm.Get_rank() == 0:
        if distributed:
            dist.destroy_process_group()
        print("\n--- All ranks finished writing. Now performing final check. ---")
        
        try:
            # final_dataset = xr.open_zarr(os.path.join(outputs_dir, f"{model_name}.zarr"), consolidated=False)
            tmp_dataset = zarr.open(os.path.join(outputs_dir, f"{model_name}.zarr"), mode='r')
            print(tmp_dataset.tree())
            final_dataset = harmonize_zarr_to_xarray(tmp_dataset)
            try:
                print(final_dataset.tree())
            except:
                print(final_dataset)
        except Exception as e:
            print(e)
    
        time_end = time.perf_counter()
        print(f"Elapsed time: {time_end - time_start:.2f} s")

if __name__ == "__main__":
    main(distributed=True)

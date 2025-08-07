import os
import shutil
from datetime import timedelta

import zarr
import numpy as np
import numcodecs

import torch
import torch.nn as nn

from mpi4py import MPI

# This is the saver logic from our previous discussion
class HourIndexedZarrSaver(nn.Module):
    # (The code for HourIndexedZarrSaver from the previous response goes here)
    # ... no changes needed to its internal logic ...
    @staticmethod
    def _td_to_hours(td: timedelta) -> int:
        """Converts a timedelta object to total integer hours."""
        return int(td.total_seconds() / 3600)
    # ... rest of the class ...
    def __init__(self, path: str, mode: str = 'a', synchronizer=None):
        super().__init__()
        self.path = path
        # Pass the synchronizer to the zarr.open_group call
        self.root = zarr.open_group(self.path, mode=mode, synchronizer=synchronizer)

        try:
            self.index_array = self.root['_index']
        except KeyError:
            self.index_array = self.root.create_dataset(
                '_index',
                shape=(0, 2),
                chunks=(1024, 2),
                dtype=object,
                object_codec=numcodecs.JSON()
            )
        # We'll print only from rank 0 to avoid spamming the log
        # print(f"HourIndexedZarrSaver initialized at '{path}'.")
        
    def __len__(self) -> int:
        return len(self.index_array)

    def __getitem__(self, idx: int) -> dict:
        if idx >= len(self):
            raise IndexError("Index out of range")
        base_time_key, lead_time_hours = self.index_array[idx]
        group = self.root[base_time_key][str(lead_time_hours)]
        outputs = {name: arr[:] for name, arr in group.arrays()}
        return {
            "base_time": base_time_key,
            "lead_time": timedelta(hours=lead_time_hours),
            "outputs": outputs,
            "attrs": dict(group.attrs)
        }

    def forward(self, sample: dict) -> dict:
        base_times = sample['base_time']
        lead_times = sample['lead_time']
        outputs_dict = sample['outputs']
        valid_times = sample.get('valid_time')
        is_batch = isinstance(base_times, (list, tuple))
        if is_batch:
            paths_to_append = []
            for i in range(len(base_times)):
                base_t_key = str(base_times[i])
                lead_t_hours = self._td_to_hours(lead_times[i])
                paths_to_append.append([base_t_key, lead_t_hours])
                lead_group = self.root.require_group(base_t_key).require_group(str(lead_t_hours))
                if valid_times:
                    lead_group.attrs['valid_time'] = str(valid_times[i])
                for name, tensor in outputs_dict.items():
                    lead_group[name] = tensor[i].detach().cpu().numpy()
            self.index_array.append(paths_to_append)
        else:
            base_t_key = str(base_times)
            lead_t_hours = self._td_to_hours(lead_times)
            lead_group = self.root.require_group(base_t_key).require_group(str(lead_t_hours))
            if valid_times:
                lead_group.attrs['valid_time'] = str(valid_times)
            for name, tensor in outputs_dict.items():
                lead_group[name] = tensor.detach().cpu().numpy()
            self.index_array.append([[base_t_key, lead_t_hours]])
        return sample


class MPIZarrSaver:
    def __init__(self, path: str):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.path = path

        # Step 1: Rank 0 cleans up and initializes the archive.
        if self.rank == 0:
            print(f"Rank 0: Initializing Zarr archive at {self.path}")
            if os.path.exists(self.path):
                shutil.rmtree(self.path)
            # Create the root directory
            os.makedirs(self.path, exist_ok=True)
        
        # Step 2: All processes must wait for rank 0 to finish.
        self.comm.Barrier()

        # Step 3: All processes open the Zarr store with a file-based lock.
        # The lock file is created inside the Zarr directory.
        lock_path = os.path.join(self.path, '.zarrlock')
        lock = zarr.ProcessSynchronizer(lock_path)
        
        self.saver = HourIndexedZarrSaver(self.path, mode='a', synchronizer=lock)
        if self.rank == 0:
            print(f"All {self.size} ranks have opened the synchronized Zarr archive.")

    def save(self, sample: dict):
        """A wrapper for the forward call to save a sample from any rank."""
        print(f"Rank {self.rank}: Saving data for base_time {sample['base_time']}")
        self.saver(sample)

    def get_saver_instance(self):
        """Returns the underlying saver object for inspection, e.g., len()."""
        return self.saver
    
# Save this script as, e.g., `run_mpi_save.py`
# And run it with: mpirun -n 4 python run_mpi_save.py

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- Setup ---
    # The path must be on a shared parallel file system accessible by all nodes.
    output_path = "mpi_output.zarr"
    saver = MPIZarrSaver(path=output_path)

    # --- Each rank prepares its own unique data to save ---
    # For example, each rank processes a different forecast initialization time.
    base_time = f"2025-08-07T{rank:02d}:00:00Z"
    sample_for_this_rank = {
        'base_time':  base_time,
        'lead_time':  timedelta(hours=6 + rank * 6), # Each rank has a different lead time
        'valid_time': f"2025-08-07T{rank+6:02d}:00:00Z",
        'outputs': {
            'data_from_rank': torch.tensor([rank] * 5),
            'size': torch.tensor(size)
        }
    }

    # --- Each rank saves its data concurrently ---
    # The lock inside the saver handles coordination automatically.
    saver.save(sample_for_this_rank)

    # --- Verification (optional, from rank 0 after all writes) ---
    comm.Barrier()
    if rank == 0:
        print("\n--- Rank 0: Verifying final archive ---")
        final_saver = saver.get_saver_instance()
        print(f"Total forecasts saved: {len(final_saver)}")
        final_saver.root.tree()

        # Check the data from the last rank
        last_rank_data = final_saver[size - 1]
        print("\nData from last rank:")
        print(last_rank_data)


if __name__ == "__main__":
    main()
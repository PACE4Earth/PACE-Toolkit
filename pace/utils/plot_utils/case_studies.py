import sys
import os
import matplotlib.pyplot as pyplot
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import UnifiedDataset

def main():
    start_time = time.perf_counter()
    config_path = "/p/project/hclimrep/vas1/PACE-Toolkit/pace/configs/config_corrdiff.json"
    model_dataset = UnifiedDataset(config_path, dataset_key="model")

    for i, (file_path, base_time, lead_idx, leadtimes, o) in enumerate(model_dataset.samples):
        valid_time = model_dataset.valid_times_for_samples[i]
        print(f"Base: {base_time}, LeadIdx: {lead_idx}, Valid: {valid_time}, File: {file_path.name}")
        sample = model_dataset[i]
        print("  base_time:", sample['base_time'])
        print("  lead_time:", sample['lead_time'])
        var_keys = [k for k in sample.keys() if k not in ['base_time', 'lead_time', 'idx']]
        for k in var_keys:
            print(f"  {k}: shape {sample[k].shape}")

    end_time = time.perf_counter()
    print(f"Elapsed time: {end_time - start_time}")

if __name__ == "__main__":
    main()
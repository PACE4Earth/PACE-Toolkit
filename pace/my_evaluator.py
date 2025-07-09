from utils.dataset import UnifiedDataset

if __name__ == "__main__":
    dataset = UnifiedDataset()

    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]

    print("Available variables:")
    for var, tensor in sample.items():
        print(f"  {var}: shape {tensor.shape}")

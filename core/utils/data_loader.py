

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config

class MetamaterialDataset(Dataset):
    """Custom PyTorch Dataset for the Metamaterial data."""
    def __init__(self, struct_params, spectra, metrics):
        self.struct_params = torch.FloatTensor(struct_params)
        self.spectra = torch.FloatTensor(spectra)
        self.metrics = torch.FloatTensor(metrics)

    def __len__(self):
        return len(self.struct_params)

    def __getitem__(self, idx):
        return {
            'struct': self.struct_params[idx],
            'spectra': self.spectra[idx],
            'metrics': self.metrics[idx]
        }

def get_dataloaders(batch_size=config.BATCH_SIZE):
    """Loads data, preprocesses it, and returns PyTorch DataLoaders."""
    # Load the dataset
    df = pd.read_csv(config.DATASET_PATH)

    # Separate features
    struct_params = df[config.STRUCT_PARAMS].values
    spectra = df[config.SPECTRA_PARAMS].values
    metrics = df[config.METRIC_PARAMS].values

    # Normalize the data
    scaler_struct = MinMaxScaler()
    struct_params_scaled = scaler_struct.fit_transform(struct_params)

    scaler_spectra = MinMaxScaler()
    spectra_scaled = scaler_spectra.fit_transform(spectra)

    scaler_metrics = MinMaxScaler()
    metrics_scaled = scaler_metrics.fit_transform(metrics)

    # Create the full dataset
    dataset = MetamaterialDataset(struct_params_scaled, spectra_scaled, metrics_scaled)

    # Split the dataset
    test_size = int(len(dataset) * config.TEST_SPLIT)
    val_size = int(len(dataset) * config.VAL_SPLIT)
    train_size = len(dataset) - test_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_STATE)
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return loaders and scalers (scalers needed for inverse transform)
    scalers = {
        'struct': scaler_struct,
        'spectra': scaler_spectra,
        'metrics': scaler_metrics
    }

    return train_loader, val_loader, test_loader, scalers

if __name__ == '__main__':
    # Example of how to use the dataloader
    train_loader, val_loader, test_loader, scalers = get_dataloaders()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # Inspect a batch
    sample_batch = next(iter(train_loader))
    print("\nSample batch shapes:")
    print(f"  Structure params: {sample_batch['struct'].shape}")
    print(f"  Spectra:          {sample_batch['spectra'].shape}")
    print(f"  Metrics:          {sample_batch['metrics'].shape}")

    # Check scaler functionality
    original_struct_sample = scalers['struct'].inverse_transform(sample_batch['struct'].numpy())
    print("\nExample of inverse-transformed structure parameters:")
    print(original_struct_sample[0])


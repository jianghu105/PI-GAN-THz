

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
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
    """Loads data, splits into train/val/test, fits scalers on train only, and returns DataLoaders and scalers."""
    # Load the dataset
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    dataset_full_path = os.path.join(project_root, config.DATASET_PATH)
    df = pd.read_csv(dataset_full_path)

    # Separate raw features (no scaling yet)
    struct_params = df[config.STRUCT_PARAMS].values
    spectra = df[config.SPECTRA_PARAMS].values
    metrics = df[config.METRIC_PARAMS].values

    # First split off test set
    struct_trainval, struct_test, spectra_trainval, spectra_test, metrics_trainval, metrics_test = train_test_split(
        struct_params, spectra, metrics,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_STATE,
        shuffle=True
    )

    # Then split train/val from trainval
    val_relative = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT) if (1.0 - config.TEST_SPLIT) > 0 else 0.0
    struct_train, struct_val, spectra_train, spectra_val, metrics_train, metrics_val = train_test_split(
        struct_trainval, spectra_trainval, metrics_trainval,
        test_size=val_relative,
        random_state=config.RANDOM_STATE,
        shuffle=True
    )

    # Fit scalers on TRAIN ONLY
    scaler_struct = MinMaxScaler().fit(struct_train)
    scaler_spectra = MinMaxScaler().fit(spectra_train)
    scaler_metrics = MinMaxScaler().fit(metrics_train)

    # Transform each split
    struct_train_scaled = scaler_struct.transform(struct_train)
    struct_val_scaled = scaler_struct.transform(struct_val)
    struct_test_scaled = scaler_struct.transform(struct_test)

    spectra_train_scaled = scaler_spectra.transform(spectra_train)
    spectra_val_scaled = scaler_spectra.transform(spectra_val)
    spectra_test_scaled = scaler_spectra.transform(spectra_test)

    metrics_train_scaled = scaler_metrics.transform(metrics_train)
    metrics_val_scaled = scaler_metrics.transform(metrics_val)
    metrics_test_scaled = scaler_metrics.transform(metrics_test)

    # Create datasets per split
    train_dataset = MetamaterialDataset(struct_train_scaled, spectra_train_scaled, metrics_train_scaled)
    val_dataset = MetamaterialDataset(struct_val_scaled, spectra_val_scaled, metrics_val_scaled)
    test_dataset = MetamaterialDataset(struct_test_scaled, spectra_test_scaled, metrics_test_scaled)

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

    # Add these lines to print statistics of scaled data
    print("\n--- Scaled Data Statistics (Train Set) ---")
    train_struct_data = torch.cat([batch['struct'] for batch in train_loader])
    train_spectra_data = torch.cat([batch['spectra'] for batch in train_loader])
    train_metrics_data = torch.cat([batch['metrics'] for batch in train_loader])

    print(f"Scaled Struct - Min: {train_struct_data.min():.4f}, Max: {train_struct_data.max():.4f}, Mean: {train_struct_data.mean():.4f}, Std: {train_struct_data.std():.4f}")
    print(f"Scaled Spectra - Min: {train_spectra_data.min():.4f}, Max: {train_spectra_data.max():.4f}, Mean: {train_spectra_data.mean():.4f}, Std: {train_spectra_data.std():.4f}")
    print(f"Scaled Metrics - Min: {train_metrics_data.min():.4f}, Max: {train_metrics_data.max():.4f}, Mean: {train_metrics_data.mean():.4f}, Std: {train_metrics_data.std():.4f}")

    print("\nScaler Scale_ values (for original range calculation):")
    print(f"Spectra Scale_: {scalers['spectra'].scale_}")
    print(f"Metrics Scale_: {scalers['metrics'].scale_}")




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import json

def preprocess_data(file_path, test_size=0.1, val_size=0.1, random_state=42):
    """
    Loads, cleans, normalizes, and splits the THz metamaterial data.

    Args:
        file_path (str): Path to the input CSV file.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the dataset to include in the validation split.
        random_state (int): Random seed for reproducibility.

    Returns:
        dict: A dictionary containing the split and structured data as NumPy arrays.
        dict: A dictionary containing the scalers for inverse transformation.
    """
    # 1. Data Loading
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None, None

    print("Successfully loaded the dataset.")
    print(f"Initial shape: {df.shape}")

    # 2. Data Cleaning and Validation
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    initial_rows = len(df)
    df.dropna(inplace=True)
    cleaned_rows = len(df)
    print(f"Removed {initial_rows - cleaned_rows} rows with NaN or infinite values.")
    print(f"Shape after cleaning: {df.shape}")

    # Identify column groups
    param_cols = ['r1', 'r2', 'w', 'g']
    feature_cols = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
    spectra_cols = [col for col in df.columns if col.startswith('Freq_')]

    # Ensure all required columns are present
    for cols in [param_cols, feature_cols]:
        for col in cols:
            if col not in df.columns:
                print(f"Error: Column '{col}' not found in the dataset.")
                return None, None
    if not spectra_cols:
        print("Error: No spectra columns (e.g., 'Freq_X.XX') found.")
        return None, None
        
    print(f"Found {len(spectra_cols)} spectra columns.")

    # Extract data groups
    structural_params = df[param_cols].values
    resonance_features = df[feature_cols].values
    transmission_spectra = df[spectra_cols].values

    # 3. Data Normalization
    param_scaler = MinMaxScaler(feature_range=(0, 1))
    spectra_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    params_normalized = param_scaler.fit_transform(structural_params)
    spectra_normalized = spectra_scaler.fit_transform(transmission_spectra)
    features_normalized = feature_scaler.fit_transform(resonance_features)
    
    print("Data normalization complete.")

    # Store scalers
    scalers = {
        'params': {'min': param_scaler.min_.tolist(), 'scale': param_scaler.scale_.tolist()},
        'spectra': {'min': spectra_scaler.min_.tolist(), 'scale': spectra_scaler.scale_.tolist()},
        'features': {'min': feature_scaler.min_.tolist(), 'scale': feature_scaler.scale_.tolist()}
    }
    
    # Save scalers to a file
    os.makedirs('config', exist_ok=True)
    with open('config/scalers.json', 'w') as f:
        json.dump(scalers, f, indent=4)
    print("Saved normalization scalers to config/scalers.json")


    # 4. Data Splitting
    # First split: separate training and temp (validation + test)
    train_indices, temp_indices = train_test_split(
        np.arange(len(df)),
        test_size=(test_size + val_size),
        random_state=random_state
    )

    # Second split: separate validation and test from temp
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=(test_size / (test_size + val_size)),
        random_state=random_state
    )
    
    # 5. Data Structuring
    data = {
        'train': {
            'params': params_normalized[train_indices],
            'spectra': spectra_normalized[train_indices],
            'features': features_normalized[train_indices]
        },
        'val': {
            'params': params_normalized[val_indices],
            'spectra': spectra_normalized[val_indices],
            'features': features_normalized[val_indices]
        },
        'test': {
            'params': params_normalized[test_indices],
            'spectra': spectra_normalized[test_indices],
            'features': features_normalized[test_indices]
        }
    }
    
    print("Data splitting and structuring complete.")

    return data, scalers

if __name__ == '__main__':
    # Define paths
    DATASET_PATH = 'dataset/THz_Metamaterial_Spectra_With_Metrics.csv'
    OUTPUT_NPZ_PATH = 'dataset/preprocessed_data.npz'

    # Process the data
    processed_data, saved_scalers = preprocess_data(DATASET_PATH)

    if processed_data:
        # 6. Output Verification
        print("\n--- Dataset Shapes ---")
        for split_name, split_data in processed_data.items():
            print(f"\n{split_name.capitalize()} Set:")
            for data_name, array in split_data.items():
                print(f"  {data_name.capitalize()}:\t{array.shape}")

        # 7. Save data to .npz file for the training script
        print(f"\nSaving processed data to {OUTPUT_NPZ_PATH}...")
        
        # Structure the data for np.savez
        save_data = {
            'train_params': processed_data['train']['params'],
            'train_spectra': processed_data['train']['spectra'],
            'train_features': processed_data['train']['features'],
            'val_params': processed_data['val']['params'],
            'val_spectra': processed_data['val']['spectra'],
            'val_features': processed_data['val']['features'],
            'test_params': processed_data['test']['params'],
            'test_spectra': processed_data['test']['spectra'],
            'test_features': processed_data['test']['features'],
        }

        # Save to a compressed .npz file
        os.makedirs(os.path.dirname(OUTPUT_NPZ_PATH), exist_ok=True)
        np.savez(OUTPUT_NPZ_PATH, **save_data)
        print(f"Successfully saved data to {OUTPUT_NPZ_PATH}")



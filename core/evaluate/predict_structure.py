# Assuming this is part of your PI_GAN_THZ/core/predict/predict_structure.py or a new utility

import argparse
import io
import numpy as np
import pandas as pd # pandas is great for reading structured text files
from scipy.interpolate import interp1d # For interpolation
import torch
import os
import sys

# Assume project_root is already added to sys.path
thz_file_path = 'dataset/THZ.txt'

# Import your existing modules
from core.models.generator import Generator
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics, normalize_metrics, normalize_spectrum
from core.utils.plot_utils import plot_fwd_model_predictions
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_mse
try:
    from tqdm.notebook import tqdm  # 适用于 Jupyter/Colab
except ImportError:
    from tqdm import tqdm  # 适用于命令行环境

def process_thz_data(thz_file_path: str, target_frequencies: np.ndarray) -> np.ndarray:
    """
    Reads THz data from a .txt file, converts S21 from dB to linear,
    and interpolates it to the target_frequencies.

    Args:
        thz_file_path (str): Path to the THZ.txt file.
        target_frequencies (np.ndarray): The array of frequencies (e.g., from MetamaterialDataset)
                                         to which the spectrum should be interpolated.

    Returns:
        np.ndarray: Interpolated linear S21 spectrum (0-1 range).
    """
    if not os.path.exists(thz_file_path):
        tqdm.write(f"Error: THz data file not found at {thz_file_path}")
        return None

    # Read the data, skipping initial comment lines
    # The actual data starts after the line starting with '#'Parameters = ...
    # and continues after the line starting with '#---------------------------'
    with open(thz_file_path, 'r') as f:
        lines = f.readlines()

    data_start_line = 0
    for i, line in enumerate(lines):
        if line.startswith('#-----------------------------'):
            data_start_line = i + 1
            break
    
    if data_start_line == 0:
        tqdm.write("Error: Could not find data start delimiter in THZ.txt.")
        return None

    data_lines = lines[data_start_line:]
    
    # Use pandas to read the tab-separated data
    df = pd.read_csv(io.StringIO("".join(data_lines)), sep='\t', header=None, names=['Frequency', 'S21_dB'])

    original_frequencies = df['Frequency'].values
    original_s21_db = df['S21_dB'].values

    # Convert S21 from dB to linear scale (Transmittance/Reflectance)
    # T = 10^(S21_dB / 10)
    linear_s21 = 10**(original_s21_db / 10.0)

    # Interpolate the linear S21 data to the target frequencies
    # Ensure target_frequencies are within the range of original_frequencies
    f_interp = interp1d(original_frequencies, linear_s21, kind='linear', fill_value="extrapolate")
    interpolated_s21 = f_interp(target_frequencies)

    # Ensure values are clamped between 0 and 1 if extrapolation yields out-of-range values
    interpolated_s21 = np.clip(interpolated_s21, 0.0, 1.0)
    
    return interpolated_s21

# --- The rest of your predict_structure_from_spectrum function (modified) ---

def predict_structure_from_spectrum(target_spectrum_data: np.ndarray, target_metrics_data: np.ndarray = None, 
                                    num_samples_to_plot: int = 1, thz_input_file: str = None):
    """
    Predicts optimal structure parameters from a target spectrum or a THz data file.

    Args:
        target_spectrum_data (np.ndarray): The target spectrum (e.g., transmittance curve).
                                           Shape: (SPECTRUM_DIM,) or (1, SPECTRUM_DIM).
                                           Assumed to be in original (denormalized) range.
                                           If thz_input_file is provided, this parameter can be None.
        target_metrics_data (np.ndarray, optional): Optional target physical metrics.
                                                    Shape: (METRICS_DIM,) or (1, METRICS_DIM).
                                                    Assumed to be in original (denormalized) range.
        num_samples_to_plot (int): Number of samples to plot (currently designed for 1).
        thz_input_file (str, optional): Path to a THZ.txt file to be processed as the target spectrum.
                                        If provided, target_spectrum_data will be ignored.
    """
    tqdm.write("\n--- Starting Structure Prediction Script ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")
    set_seed(cfg.RANDOM_SEED)
    tqdm.write(f"Random seed set to: {cfg.RANDOM_SEED}")
    cfg.create_directories()
    tqdm.write(f"Plots will be saved to: {cfg.PLOTS_DIR}")

    # --- Data Dummy Loading for Normalization Ranges ---
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        tqdm.write(f"Error: Dataset for normalization ranges not found at {data_path}.")
        return None, None
    
    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    tqdm.write(f"Dataset loaded to retrieve normalization ranges and metadata.")
    
    frequencies = dataset.frequencies # This is the target_frequencies for interpolation
    metric_names = dataset.metric_names

    # --- Model Initialization and Loading (same as before) ---
    generator = Generator(input_dim=cfg.SPECTRUM_DIM, output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM).to(device)
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    gen_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth")
    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth") 

    models_found = True
    if not os.path.exists(gen_model_path):
        tqdm.write(f"Error: Generator model not found at {gen_model_path}.")
        models_found = False
    if not os.path.exists(fwd_model_path):
        tqdm.write(f"Error: ForwardModel not found at {fwd_model_path}.")
        models_found = False
    
    if not models_found:
        tqdm.write("Please ensure PI-GAN training and ForwardModel training are complete and models are saved.")
        return None, None

    try:
        generator.load_state_dict(torch.load(gen_model_path, map_location=device))
        forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device))
        tqdm.write(f"Generator and ForwardModel loaded from {cfg.SAVED_MODELS_DIR}.")
    except Exception as e:
        tqdm.write(f"Error: An exception occurred while loading models: {e}")
        tqdm.write("Please check if model files are corrupted or do not match the architecture.")
        return None, None

    generator.eval() 
    forward_model.eval() 

    # --- Prepare Input Data ---
    # Determine the actual target spectrum data based on input arguments
    final_target_spectrum_data = None
    if thz_input_file:
        tqdm.write(f"Processing THz data from {thz_input_file}...")
        final_target_spectrum_data = process_thz_data(thz_input_file, frequencies) # Use dataset.frequencies as target
        if final_target_spectrum_data is None:
            return None, None
        tqdm.write("THZ data processed successfully.")
    elif target_spectrum_data is not None:
        final_target_spectrum_data = target_spectrum_data
    else:
        tqdm.write("Error: No target spectrum provided. Please specify --spectrum_path or --thz_input_file.")
        return None, None

    # Ensure final_target_spectrum_data has correct shape (1, SPECTRUM_DIM)
    if final_target_spectrum_data.ndim == 1:
        final_target_spectrum_data = final_target_spectrum_data[np.newaxis, :] # Add batch dimension
    if final_target_spectrum_data.shape[1] != cfg.SPECTRUM_DIM:
        tqdm.write(f"Error: Input target spectrum dimension mismatch after processing. Expected {cfg.SPECTRUM_DIM}, got {final_target_spectrum_data.shape[1]}.")
        return None, None

    # Normalize the target spectrum
    target_spectrum_norm = normalize_spectrum(torch.tensor(final_target_spectrum_data, dtype=torch.float32), 
                                              dataset.spectrum_min, dataset.spectrum_max).to(device)
    
    # Prepare target metrics (if provided)
    target_metrics_norm = None
    if target_metrics_data is not None:
        if target_metrics_data.ndim == 1:
            target_metrics_data = target_metrics_data[np.newaxis, :]
        if target_metrics_data.shape[1] != cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM:
             tqdm.write(f"Warning: Input target metrics dimension mismatch. Expected {cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM}, got {target_metrics_data.shape[1]}. Skipping metrics comparison.")
             target_metrics_data = None
        else:
            target_metrics_norm = normalize_metrics(torch.tensor(target_metrics_data, dtype=torch.float32),
                                                    dataset.metric_mins, dataset.metric_maxs).to(device)


    # --- Prediction Process (same as before) ---
    tqdm.write("\n--- Performing Inverse Prediction ---")
    with torch.no_grad():
        predicted_params_norm = generator(target_spectrum_norm)
        predicted_params_denorm = denormalize_params(predicted_params_norm, dataset.param_ranges).cpu().numpy()
        
        tqdm.write(f"Predicted Denormalized Parameters: {predicted_params_denorm[0]}")

        recon_spectrum_norm, recon_metrics_norm = forward_model(predicted_params_norm)
        
        recon_spectrum_np = recon_spectrum_norm.cpu().numpy()
        recon_metrics_denorm_np = denormalize_metrics(recon_metrics_norm, dataset.metric_ranges).cpu().numpy()

        mse_criterion = criterion_mse()
        spectrum_mse = mse_criterion(recon_spectrum_norm, target_spectrum_norm).item()
        tqdm.write(f"Spectrum Reconstruction MSE (Target vs. Reconstructed by Fwd Model): {spectrum_mse:.6f}")

        metrics_mse = float('nan')
        if target_metrics_norm is not None:
            metrics_mse = mse_criterion(recon_metrics_norm, target_metrics_norm).item()
            tqdm.write(f"Metrics Reconstruction MSE (Target vs. Reconstructed by Fwd Model): {metrics_mse:.6f}")
        else:
            tqdm.write("Target metrics not provided or dimension mismatch, skipping metrics MSE calculation.")


    # --- Plotting Results ---
    tqdm.write("\n--- Generating Prediction Verification Plot ---")
    
    plot_fwd_model_predictions(
        real_params=predicted_params_denorm, 
        real_spectrums=final_target_spectrum_data, 
        predicted_spectrums=recon_spectrum_np, 
        real_metrics=target_metrics_data if target_metrics_data is not None else np.zeros((1, cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM)),
        predicted_metrics=recon_metrics_denorm_np, 
        frequencies=frequencies,
        num_samples=num_samples_to_plot,
        save_path=os.path.join(cfg.PLOTS_DIR, 'predicted_structure_verification.png'),
        metric_names=metric_names
    )
    tqdm.write(f"Prediction verification plot saved to {cfg.PLOTS_DIR}")

    tqdm.write("--- Structure Prediction Script Completed ---")
    
    return predicted_params_denorm[0], {"spectrum_mse": spectrum_mse, "metrics_mse": metrics_mse}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predicts optimal structure parameters from a target spectrum or THz data.")
    parser.add_argument('--spectrum_path', type=str, default=None,
                        help='Path to a .npy file containing the target spectrum (denormalized).')
    parser.add_argument('--metrics_path', type=str, default=None,
                        help='(Optional) Path to a .npy file containing the target metrics (denormalized).')
    parser.add_argument('--thz_input_file', type=str, default=None,
                        help='Path to the raw THZ.txt file for processing into a target spectrum.')
    parser.add_argument('--num_plot_samples', type=int, default=1,
                        help='Number of samples to plot (default: 1).')
    args = parser.parse_args()

    # Priority: thz_input_file > spectrum_path
    if not args.thz_input_file and not args.spectrum_path:
        print("Error: Either --spectrum_path or --thz_input_file must be provided.")
        sys.exit(1)

    target_spectrum = None
    if args.spectrum_path and not args.thz_input_file:
        if os.path.exists(args.spectrum_path):
            target_spectrum = np.load(args.spectrum_path)
            print(f"Loaded target spectrum from: {args.spectrum_path}, shape: {target_spectrum.shape}")
        else:
            print(f"Error: Target spectrum .npy file not found at {args.spectrum_path}.")
            sys.exit(1)
            
    target_metrics = None
    if args.metrics_path:
        if os.path.exists(args.metrics_path):
            target_metrics = np.load(args.metrics_path)
            print(f"Loaded target metrics from: {args.metrics_path}, shape: {target_metrics.shape}")
        else:
            print(f"Warning: Target metrics .npy file not found at {args.metrics_path}. Proceeding without target metrics comparison.")


    # Call the main prediction function
    predicted_params, evaluation_results = predict_structure_from_spectrum(
        target_spectrum_data=target_spectrum, # This will be None if thz_input_file is provided
        target_metrics_data=target_metrics,
        num_samples_to_plot=args.num_plot_samples,
        thz_input_file=args.thz_input_file # This will trigger the parsing if provided
    )

    if predicted_params is not None:
        print("\n--- Final Predicted Structure Parameters ---")
        print(f"Parameters: {predicted_params}")
        print(f"Evaluation MSEs: {evaluation_results}")
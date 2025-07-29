

import torch
import numpy as np
import json
import argparse
from sklearn.preprocessing import MinMaxScaler

from core.models.generator import Generator

def evaluate(args):
    """
    Evaluates the trained Generator model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load test data
    try:
        with np.load(args.data_path) as data:
            test_spectra = data['test_spectra']
            test_params_original = data['test_params'] # These are normalized
    except FileNotFoundError:
        print(f"Error: Preprocessed data not found at {args.data_path}.")
        return

    # Load scalers
    try:
        with open(args.scalers_path, 'r') as f:
            scalers_data = json.load(f)
        param_scaler = MinMaxScaler()
        param_scaler.min_ = np.array(scalers_data['params']['min'])
        param_scaler.scale_ = np.array(scalers_data['params']['scale'])
    except FileNotFoundError:
        print(f"Error: Scalers not found at {args.scalers_path}.")
        return

    # Initialize Generator
    generator = Generator(spectra_dim=args.spectra_dim, params_dim=args.params_dim, noise_dim=args.noise_dim).to(device)
    
    # Load trained generator weights
    try:
        generator.load_state_dict(torch.load(args.generator_path, map_location=device))
        generator.eval()
        print(f"Successfully loaded trained generator from {args.generator_path}")
    except FileNotFoundError:
        print(f"Error: Generator model not found at {args.generator_path}.")
        return

    # Select a few samples for evaluation
    num_samples = 5
    sample_indices = np.random.choice(len(test_spectra), num_samples, replace=False)
    
    sample_spectra = torch.from_numpy(test_spectra[sample_indices]).float().to(device)
    sample_params_normalized = test_params_original[sample_indices]

    # Generate parameters
    with torch.no_grad():
        noise = torch.randn(num_samples, args.noise_dim, device=device)
        generated_params_normalized = generator(sample_spectra, noise).cpu().numpy()

    # Inverse transform to get original scale
    generated_params_original = param_scaler.inverse_transform(generated_params_normalized)
    true_params_original = param_scaler.inverse_transform(sample_params_normalized)

    # Print results
    print("\n--- Evaluation Results ---")
    for i in range(num_samples):
        print(f"\nSample {i+1}:")
        print(f"  - True Params (Original Scale):     {np.round(true_params_original[i], 4)}")
        print(f"  - Generated Params (Original Scale): {np.round(generated_params_original[i], 4)}")
        
        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((true_params_original[i] - generated_params_original[i]) / true_params_original[i])) * 100
        print(f"  - MAPE: {mape:.2f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the trained PI-GAN Generator.")
    
    parser.add_argument("--generator_path", type=str, default="saved_models/generator_epoch_150.pth", help="Path to the trained generator model.")
    parser.add_argument("--data_path", type=str, default="dataset/preprocessed_data.npz", help="Path to the preprocessed .npz data file.")
    parser.add_argument("--scalers_path", type=str, default="config/scalers.json", help="Path to the scalers JSON file.")
    parser.add_argument("--use_cuda", type=bool, default=True, help="Whether to use CUDA if available.")
    parser.add_argument("--spectra_dim", type=int, default=250, help="Dimensionality of the spectra data.")
    parser.add_argument("--params_dim", type=int, default=4, help="Dimensionality of the structural parameters.")
    parser.add_argument("--noise_dim", type=int, default=100, help="Dimensionality of the noise vector.")

    args = parser.parse_args()
    evaluate(args)

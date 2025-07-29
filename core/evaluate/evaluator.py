import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.models.generator import Generator
from core.models.forward_model import ForwardModel
from core.utils.data_loader import get_dataloaders

def evaluate_model(num_samples=1000):
    """Evaluates the trained PI-GAN model by generating samples and analyzing them."""
    print("Starting model evaluation...")

    # Load scalers (needed for inverse transform)
    _, _, _, scalers = get_dataloaders(batch_size=config.BATCH_SIZE)

    # Load trained Generator
    generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS)
    ).to(config.DEVICE)
    generator_path = os.path.join(config.SAVED_MODELS_DIR, 'final_generator.pth')
    if not os.path.exists(generator_path):
        # Try to find the latest saved generator checkpoint
        generator_checkpoints = [f for f in os.listdir(config.SAVED_MODELS_DIR) if f.startswith('generator_epoch_') and f.endswith('.pth')]
        if not generator_checkpoints:
            print(f"Error: No generator model found in {config.SAVED_MODELS_DIR}. Please train the GAN first.")
            return
        generator_checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
        generator_path = os.path.join(config.SAVED_MODELS_DIR, generator_checkpoints[0])
        print(f"Using latest generator checkpoint: {generator_path}")
    generator.load_state_dict(torch.load(generator_path))
    generator.eval()

    # Load pre-trained Forward Model (for predicting spectra/metrics of generated structures)
    forward_model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS)
    ).to(config.DEVICE)
    forward_model_path = os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')
    if not os.path.exists(forward_model_path):
        print(f"Error: Forward model not found at {forward_model_path}. Please pre-train the forward model first.")
        return
    forward_model.load_state_dict(torch.load(forward_model_path))
    forward_model.eval()

    # Generate samples
    generated_structs = []
    generated_spectra = []
    generated_metrics = []

    with torch.no_grad():
        for _ in range(num_samples // config.BATCH_SIZE):
            z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
            fake_struct_normalized = generator(z)
            fake_spectra_normalized, fake_metrics_normalized = forward_model(fake_struct_normalized)

            generated_structs.append(fake_struct_normalized.cpu().numpy())
            generated_spectra.append(fake_spectra_normalized.cpu().numpy())
            generated_metrics.append(fake_metrics_normalized.cpu().numpy())

    generated_structs = np.vstack(generated_structs)
    generated_spectra = np.vstack(generated_spectra)
    generated_metrics = np.vstack(generated_metrics)

    # Inverse transform to original scale
    generated_structs_original = scalers['struct'].inverse_transform(generated_structs)
    generated_spectra_original = scalers['spectra'].inverse_transform(generated_spectra)
    generated_metrics_original = scalers['metrics'].inverse_transform(generated_metrics)

    # Convert to DataFrames for easier analysis
    df_generated_structs = pd.DataFrame(generated_structs_original, columns=config.STRUCT_PARAMS)
    df_generated_spectra = pd.DataFrame(generated_spectra_original, columns=config.SPECTRA_PARAMS)
    df_generated_metrics = pd.DataFrame(generated_metrics_original, columns=config.METRIC_PARAMS)

    print("\n--- Generated Structural Parameters Summary ---")
    print(df_generated_structs.describe())

    print("\n--- Generated Metrics Summary ---")
    print(df_generated_metrics.describe())

    # Optional: Save generated data to CSV for further analysis
    import tempfile

    output_dir = tempfile.mkdtemp(prefix="generated_data_")
    print(f"Generated data will be saved to temporary directory: {output_dir}/")
    # os.makedirs(output_dir, exist_ok=True) # mkdtemp already creates the directory
    df_generated_structs.to_csv(os.path.join(output_dir, "generated_structs.csv"), index=False)
    df_generated_spectra.to_csv(os.path.join(output_dir, "generated_spectra.csv"), index=False)
    df_generated_metrics.to_csv(os.path.join(output_dir, "generated_metrics.csv"), index=False)
    print(f"\nGenerated data saved to {output_dir}/")

    # Load real data for comparison
    df_real = pd.read_csv(config.DATASET_PATH)

    # --- Evaluate Forward Model Prediction Accuracy ---
    print("\n--- Evaluating Forward Model Prediction Accuracy ---")
    _, _, test_loader, _ = get_dataloaders(batch_size=config.BATCH_SIZE)
    
    forward_model_mse = 0.0
    forward_model_mae = 0.0
    num_test_samples = 0
    
    mse_criterion = torch.nn.MSELoss(reduction='sum')
    mae_criterion = torch.nn.L1Loss(reduction='sum')

    with torch.no_grad():
        for batch in test_loader:
            struct_params = batch['struct'].to(config.DEVICE)
            target_spectra = batch['spectra'].to(config.DEVICE)
            
            predicted_spectra, _ = forward_model(struct_params)
            
            forward_model_mse += mse_criterion(predicted_spectra, target_spectra).item()
            forward_model_mae += mae_criterion(predicted_spectra, target_spectra).item()
            num_test_samples += struct_params.shape[0]

    avg_forward_model_mse = forward_model_mse / num_test_samples
    avg_forward_model_mae = forward_model_mae / num_test_samples

    print(f"Forward Model Test MSE: {avg_forward_model_mse:.6f}")
    print(f"Forward Model Test MAE: {avg_forward_model_mae:.6f}")

    # Plot distributions for all structural parameters
    print("\nPlotting structural parameter distributions...")
    for param in config.STRUCT_PARAMS:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_real[param], color='blue', label='Real', kde=True, stat='density', alpha=0.5)
        sns.histplot(df_generated_structs[param], color='red', label='Generated', kde=True, stat='density', alpha=0.5)
        plt.title(f'Distribution of {param}')
        plt.xlabel(param)
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{param}_distribution.png'))
        plt.close()
        print(f"Distribution plot for {param} saved to {output_dir}/{param}_distribution.png")

    # Plot distributions for all metric parameters
    print("\nPlotting metric parameter distributions...")
    for param in config.METRIC_PARAMS:
        plt.figure(figsize=(10, 6))
        sns.histplot(df_real[param], color='blue', label='Real', kde=True, stat='density', alpha=0.5)
        sns.histplot(df_generated_metrics[param], color='red', label='Generated', kde=True, stat='density', alpha=0.5)
        plt.title(f'Distribution of {param}')
        plt.xlabel(param)
        plt.ylabel('Density')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{param}_distribution.png'))
        plt.close()
        print(f"Distribution plot for {param} saved to {output_dir}/{param}_distribution.png")

    # Plot generated spectra vs real spectra
    print("\nPlotting sample spectra...")
    num_spectra_to_plot = 5
    real_spectra_indices = np.random.choice(len(df_real), num_spectra_to_plot, replace=False)
    generated_spectra_indices = np.random.choice(len(df_generated_spectra), num_spectra_to_plot, replace=False)

    freq_values = np.linspace(float(config.SPECTRA_PARAMS[0].split('_')[1]), float(config.SPECTRA_PARAMS[-1].split('_')[1]), len(config.SPECTRA_PARAMS)).astype(float)

    plt.figure(figsize=(12, 8))
    for i in range(num_spectra_to_plot):
        plt.plot(freq_values, df_real.iloc[real_spectra_indices[i]][config.SPECTRA_PARAMS], label=f'Real {i+1}', alpha=0.7)
        plt.plot(freq_values, df_generated_spectra.iloc[generated_spectra_indices[i]], '--', label=f'Generated {i+1}', alpha=0.7)

    plt.title('Comparison of Real and Generated Spectra')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Magnitude (dB)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'spectra_comparison.png'))
    plt.close()
    print(f"Spectra comparison plot saved to {output_dir}/spectra_comparison.png")

    print("Model evaluation complete.")

if __name__ == '__main__':
    evaluate_model()
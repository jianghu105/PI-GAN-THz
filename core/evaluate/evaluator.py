import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.models.generator import Generator
from core.models.forward_model import ForwardModel
from core.utils.data_loader import get_dataloaders

def evaluate_model(num_samples=1000, condition_source="real", target_metrics_csv=None):
    print("Starting model evaluation...")
    print(f"Configured GENERATED_DATA_DIR: {config.GENERATED_DATA_DIR}")
    
    # Ensure output directory exists
    output_dir = config.GENERATED_DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    try:
        # Load scalers (needed for inverse transform)
        print("Loading dataloaders and scalers...")
        _, _, _, scalers = get_dataloaders(batch_size=config.BATCH_SIZE)
        print("Dataloaders and scalers loaded.")

        # Load trained Generator
        print("Loading Generator...")
        generator = Generator(
            latent_dim=config.LATENT_DIM,
            output_dim=len(config.STRUCT_PARAMS),
            condition_dim=config.CONDITION_DIM
        ).to(config.DEVICE)
        generator_path = os.path.join(config.SAVED_MODELS_DIR, 'final_generator.pth')
        if not os.path.exists(generator_path):
            generator_checkpoints = [f for f in os.listdir(config.SAVED_MODELS_DIR) if f.startswith('generator_epoch_') and f.endswith('.pth')]
            if not generator_checkpoints:
                print(f"Error: No generator model found in {config.SAVED_MODELS_DIR}. Please train the GAN first.")
                return
            generator_checkpoints.sort(key=lambda x: int(x.split('_')[2].split('.')[0]), reverse=True)
            generator_path = os.path.join(config.SAVED_MODELS_DIR, generator_checkpoints[0])
            print(f"Using latest generator checkpoint: {generator_path}")
        generator.load_state_dict(torch.load(generator_path, map_location=torch.device('cpu')))
        generator.eval()
        print("Generator loaded successfully.")

        # Load pre-trained Forward Model
        print("Loading Forward Model...")
        forward_model = ForwardModel(
            input_dim=len(config.STRUCT_PARAMS),
            output_dim=len(config.SPECTRA_PARAMS)
        ).to(config.DEVICE)
        forward_model_path = os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')
        if not os.path.exists(forward_model_path):
            print(f"Error: Forward model not found at {forward_model_path}. Please pre-train the forward model first.")
            return
        forward_model.load_state_dict(torch.load(forward_model_path, map_location=torch.device('cpu')))
        forward_model.eval()
        print("Forward Model loaded successfully.")

        # Generate samples
        generated_structs = []
        generated_spectra = []
        generated_metrics = []

        print(f"Generating {num_samples} samples...")
        num_batches = num_samples // config.BATCH_SIZE
        if num_batches == 0:
            print(f"Warning: num_samples ({num_samples}) is less than BATCH_SIZE ({config.BATCH_SIZE}). No samples will be generated.")
            print("Please ensure num_samples is sufficient or BATCH_SIZE is adjusted.")
            return # Exit if no samples will be generated

        # Prepare conditional vectors
        if condition_source == "real":
            # Sample conditions from REAL metrics distribution (normalized via train-fitted scaler)
            print("Condition source: real dataset metrics distribution")
            df_all = pd.read_csv(config.DATASET_PATH)
            metrics_pool_raw = df_all[config.METRIC_PARAMS].values
            metrics_pool_normalized = scalers['metrics'].transform(metrics_pool_raw)
        elif condition_source == "csv":
            if target_metrics_csv is None or not os.path.exists(target_metrics_csv):
                print(f"Error: target_metrics_csv not provided or not found: {target_metrics_csv}")
                return
            print(f"Condition source: user CSV at {target_metrics_csv}")
            df_user = pd.read_csv(target_metrics_csv)
            # Expect columns to include config.METRIC_PARAMS; allow extra columns
            missing = [c for c in config.METRIC_PARAMS if c not in df_user.columns]
            if missing:
                print(f"Error: CSV missing required metric columns: {missing}")
                return
            metrics_pool_raw = df_user[config.METRIC_PARAMS].values
            metrics_pool_normalized = scalers['metrics'].transform(metrics_pool_raw)
        else:
            print(f"Error: Unknown condition_source '{condition_source}'. Use 'real' or 'csv'.")
            return

        with torch.no_grad():
            for i in range(num_batches):
                z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
                # Draw a random minibatch of normalized metrics as condition
                idx = np.random.choice(metrics_pool_normalized.shape[0], size=config.BATCH_SIZE, replace=True)
                condition_for_G_np = metrics_pool_normalized[idx]
                condition_for_G = torch.from_numpy(condition_for_G_np).float().to(config.DEVICE)

                fake_struct_normalized = generator(z, condition_for_G)
                fake_spectra_normalized, fake_metrics_normalized = forward_model(fake_struct_normalized)

                generated_structs.append(fake_struct_normalized.cpu().numpy())
                generated_spectra.append(fake_spectra_normalized.cpu().numpy())
                generated_metrics.append(fake_metrics_normalized.cpu().numpy())
            print(f"Successfully generated {num_batches * config.BATCH_SIZE} samples.")

        generated_structs = np.vstack(generated_structs)
        generated_spectra = np.vstack(generated_spectra)
        generated_metrics = np.vstack(generated_metrics)

        # Inverse transform to original scale
        print("Performing inverse transformations...")
        generated_structs_original = scalers['struct'].inverse_transform(generated_structs)
        generated_spectra_original = scalers['spectra'].inverse_transform(generated_spectra)
        generated_metrics_original = scalers['metrics'].inverse_transform(generated_metrics)
        print("Inverse transformations complete.")

        # Convert to DataFrames for easier analysis
        df_generated_structs = pd.DataFrame(generated_structs_original, columns=config.STRUCT_PARAMS)
        df_generated_spectra = pd.DataFrame(generated_spectra_original, columns=config.SPECTRA_PARAMS)
        df_generated_metrics = pd.DataFrame(generated_metrics_original, columns=config.METRIC_PARAMS)

        print("\n--- Generated Structural Parameters Summary ---")
        print(df_generated_structs.describe())

        print("\n--- Generated Metrics Summary ---")
        print(df_generated_metrics.describe())

        # Conditional consistency: how close generated metrics are to conditions used
        if condition_source == "real":
            # Reconstruct the normalized conditions used for each batch to compare
            # For simplicity, estimate consistency by passing generated spectra through forward model metrics (already done)
            # and comparing distribution with real dataset metrics
            df_real_metrics = pd.DataFrame(pd.read_csv(config.DATASET_PATH)[config.METRIC_PARAMS], columns=config.METRIC_PARAMS)
            print("\n--- Conditional Consistency (Generated vs Real Metrics) ---")
            consistency_mae = (df_generated_metrics - df_real_metrics.sample(n=len(df_generated_metrics), replace=True).reset_index(drop=True)).abs().mean()
            print(consistency_mae)
        elif condition_source == "csv" and target_metrics_csv is not None:
            df_target = pd.read_csv(target_metrics_csv)
            # Align columns
            df_target = df_target[config.METRIC_PARAMS]
            # Broadcast/trim to match lengths
            if len(df_target) < len(df_generated_metrics):
                df_target = pd.concat([df_target] * (int(np.ceil(len(df_generated_metrics)/len(df_target)))), ignore_index=True)[:len(df_generated_metrics)]
            elif len(df_target) > len(df_generated_metrics):
                df_target = df_target.iloc[:len(df_generated_metrics)].reset_index(drop=True)
            print("\n--- Conditional Consistency (Generated vs Target CSV Metrics) ---")
            consistency_mae = (df_generated_metrics - df_target).abs().mean()
            print(consistency_mae)

        # Save generated data to CSV
        print(f"Saving generated data to: {output_dir}/")
        df_generated_structs.to_csv(os.path.join(output_dir, "generated_structs.csv"), index=False)
        df_generated_spectra.to_csv(os.path.join(output_dir, "generated_spectra.csv"), index=False)
        df_generated_metrics.to_csv(os.path.join(output_dir, "generated_metrics.csv"), index=False)
        print(f"Generated data saved to {output_dir}/")

        # Load real data for comparison
        print("Loading real data for comparison...")
        df_real = pd.read_csv(config.DATASET_PATH)
        print("Real data loaded.")

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

        # Plot distributions
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

    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate PI-GAN generator and forward model")
    parser.add_argument('--num-samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--condition', type=str, default='real', choices=['real', 'csv'], help='Condition source: real dataset metrics or a CSV file')
    parser.add_argument('--target-metrics-csv', type=str, default=None, help='Path to CSV containing target metrics columns')
    args = parser.parse_args()

    evaluate_model(num_samples=args.num_samples, condition_source=args.condition, target_metrics_csv=args.target_metrics_csv)
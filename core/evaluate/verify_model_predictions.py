import sys
import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, Subset # Ensure DataLoader and Subset are imported
try:
    from tqdm.notebook import tqdm  # 适用于 Jupyter/Colab
except ImportError:
    from tqdm import tqdm  # 适用于命令行环境

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import models, data loader, and plotting utilities
from core.models.generator import Generator
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.plot_utils import plot_fwd_model_predictions # Reuse forward model prediction plotting function
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_mse # Used for calculating MSE

def verify_predictions(num_samples: int = 10):
    """
    Verifies the prediction capabilities of the models, including
    Generator's inverse prediction and ForwardModel's forward prediction.

    Args:
        num_samples (int): The number of samples whose prediction results
                           will be verified and visualized.
    """
    tqdm.write("\n--- Starting Model Prediction Verification Script ---")
    sys.stdout.flush() # Ensure immediate output

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}")
    sys.stdout.flush()

    # Set random seed for reproducibility
    set_seed(cfg.RANDOM_SEED)
    tqdm.write(f"Random seed set to: {cfg.RANDOM_SEED}")
    sys.stdout.flush()

    # Ensure necessary directories are created, including plots directory
    cfg.create_directories()
    tqdm.write(f"Plots will be saved to: {cfg.PLOTS_DIR}")
    sys.stdout.flush()

    # --- Data Loading ---
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        tqdm.write(f"Error: Dataset not found at {data_path}. Please check config.py and ensure the CSV file exists.")
        sys.stdout.flush() # Ensure immediate output
        return # Gracefully exit

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    tqdm.write(f"Dataset loaded, containing {len(dataset)} samples.")
    sys.stdout.flush()

    # --- Model Initialization and Loading ---
    generator = Generator(input_dim=cfg.SPECTRUM_DIM, output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM).to(device)
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    # Define model save paths
    gen_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth")
    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth") 

    # Check if model files exist
    models_found = True
    if not os.path.exists(gen_model_path):
        tqdm.write(f"Error: Generator model not found at {gen_model_path}.")
        sys.stdout.flush()
        models_found = False
    if not os.path.exists(fwd_model_path):
        tqdm.write(f"Error: ForwardModel not found at {fwd_model_path}.")
        sys.stdout.flush()
        models_found = False
    
    if not models_found:
        tqdm.write("Please ensure PI-GAN training and ForwardModel pre-training are complete and all models are saved.")
        sys.stdout.flush()
        return

    try:
        generator.load_state_dict(torch.load(gen_model_path, map_location=device))
        forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device))
        tqdm.write(f"Generator and ForwardModel loaded from {cfg.SAVED_MODELS_DIR}.")
        sys.stdout.flush()
    except Exception as e:
        tqdm.write(f"Error: An exception occurred while loading models: {e}")
        tqdm.write("Please check if model files are corrupted or do not match the model architecture.")
        sys.stdout.flush()
        return

    generator.eval() # Switch to evaluation mode
    forward_model.eval() # Switch to evaluation mode

    # --- Randomly select samples for prediction and visualization ---
    tqdm.write(f"\n--- Generating Model Prediction Visualizations for {num_samples} samples ---")
    sys.stdout.flush()
    mse_criterion = criterion_mse() # Initialize MSE criterion

    if num_samples <= 0:
        tqdm.write("Warning: Number of samples to verify is 0 or negative, skipping prediction verification.")
        sys.stdout.flush()
        return
    
    if num_samples > len(dataset):
        tqdm.write(f"Warning: Number of samples to verify ({num_samples}) exceeds dataset size ({len(dataset)}). Verifying all {len(dataset)} samples.")
        sys.stdout.flush()
        num_samples = len(dataset)
    
    if num_samples == 0:
        tqdm.write("Not enough samples available for verification.")
        sys.stdout.flush()
        return

    # Randomly select sample indices
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset_dataset = Subset(dataset, sample_indices)
    # Use DataLoader to load only the selected samples, even if it's just one batch
    sample_dataloader = DataLoader(subset_dataset, batch_size=num_samples, shuffle=False)
    
    # Get the first (and only) batch of data
    # Unpack all components, including normalized versions for MSE calculation
    real_spectrums_batch, real_params_denorm_batch, real_params_norm_batch, \
    real_metrics_denorm_batch, real_metrics_norm_batch = next(iter(sample_dataloader))

    # Move real data to device
    real_spectrums = real_spectrums_batch.to(device)
    real_params_denorm_for_plot = real_params_denorm_batch.to(device) # Real unnormalized params for plotting
    real_metrics_denorm_for_plot = real_metrics_denorm_batch.to(device) # Real unnormalized metrics for plotting
    real_metrics_norm_for_mse = real_metrics_norm_batch.to(device) # Real normalized metrics for MSE

    total_spectrum_mse = 0.0
    total_metrics_mse = 0.0
    num_processed_samples = 0

    with torch.no_grad():
        # 1. Use Generator to predict parameters from real spectra
        # Input to Generator is normalized real spectrum
        predicted_params_norm = generator(real_spectrums)
        # Denormalize predicted parameters for plotting
        predicted_params_denorm = denormalize_params(predicted_params_norm, dataset.param_ranges).cpu().numpy()

        # 2. Use ForwardModel to predict spectra and metrics from the predicted parameters
        predicted_spectrum, predicted_metrics_norm = forward_model(predicted_params_norm)
        # Denormalize predicted spectrum and metrics for plotting
        predicted_spectrum_np = predicted_spectrum.cpu().numpy()
        predicted_metrics_denorm_np = denormalize_metrics(predicted_metrics_norm, dataset.metric_ranges).cpu().numpy()

        # Calculate MSE (Real Spectrum vs. Forward Model Predicted Spectrum)
        # Here we compare the original real_spectrums (normalized) with the reconstructed one.
        total_spectrum_mse += mse_criterion(predicted_spectrum, real_spectrums).item() * num_samples
        
        # Calculate MSE for metrics (Real Normalized Metrics vs. Forward Model Predicted Normalized Metrics)
        total_metrics_mse += mse_criterion(predicted_metrics_norm, real_metrics_norm_for_mse).item() * num_samples

        num_processed_samples += num_samples

        # Collect all sample data for plotting (already correctly shaped due to batching)
        all_real_params = real_params_denorm_for_plot.cpu().numpy()
        all_real_spectrums = real_spectrums.cpu().numpy()
        all_predicted_spectrums = predicted_spectrum_np 
        all_real_metrics = real_metrics_denorm_for_plot.cpu().numpy()
        all_predicted_metrics = predicted_metrics_denorm_np 

    # --- Plotting Visualization Results ---
    frequencies = dataset.frequencies # Get frequencies from MetamaterialDataset
    metric_names = dataset.metric_names # Get metric names from MetamaterialDataset

    if all_real_params.size > 0: # Ensure there's data to plot
        plot_fwd_model_predictions(
            real_params=all_real_params,
            real_spectrums=all_real_spectrums,
            predicted_spectrums=all_predicted_spectrums,
            real_metrics=all_real_metrics,
            predicted_metrics=all_predicted_metrics,
            frequencies=frequencies,
            num_samples=num_samples, # Use the actual number of processed samples
            save_path=os.path.join(cfg.PLOTS_DIR, 'model_prediction_verification.png'),
            metric_names=metric_names
        )
        tqdm.write(f"Model prediction verification plot saved to {cfg.PLOTS_DIR}")
        sys.stdout.flush()
    else:
        tqdm.write("No prediction verification data available for plotting.")
        sys.stdout.flush()

    if num_processed_samples > 0:
        avg_spectrum_mse = total_spectrum_mse / num_processed_samples
        avg_metrics_mse = total_metrics_mse / num_processed_samples
        tqdm.write(f"\nAverage Spectrum MSE (Real vs. Forward Model Predicted): {avg_spectrum_mse:.4f}")
        tqdm.write(f"Average Metrics MSE (Real vs. Forward Model Predicted): {avg_metrics_mse:.4f}")
        sys.stdout.flush()
    else:
        tqdm.write("No samples processed for MSE calculation.")
        sys.stdout.flush()

    tqdm.write("--- Model Prediction Verification Script Completed ---")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the prediction capabilities of PI-GAN Generator and ForwardModel.")
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to verify and visualize predictions for (default: 10)')
    args = parser.parse_args()
    
    verify_predictions(num_samples=args.num_samples)
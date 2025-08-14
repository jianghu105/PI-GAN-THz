import torch
import torch.nn as nn
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.models.forward_model import ForwardModel
from core.utils.data_loader import get_dataloaders

def evaluate_forward_model_standalone():
    print("Starting standalone forward model evaluation...")

    # Load DataLoaders and scalers
    _, _, test_loader, scalers = get_dataloaders(batch_size=config.BATCH_SIZE)

    # Initialize and load the Forward Model
    model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS)
    ).to(config.DEVICE)

    model_path = os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Forward model not found at {model_path}. Please pre-train the forward model first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval() # Set to evaluation mode
    print(f"Forward model loaded from {model_path}")

    # Loss criteria (sum over all samples, average later)
    spectra_mse_criterion = nn.MSELoss(reduction='sum')
    spectra_mae_criterion = nn.L1Loss(reduction='sum')
    metrics_mse_criterion = nn.MSELoss(reduction='sum')
    metrics_mae_criterion = nn.L1Loss(reduction='sum')

    # Prepare scaler tensors for metrics normalization alignment
    metrics_scale = torch.tensor(scalers['metrics'].scale_, dtype=torch.float32).to(config.DEVICE)
    metrics_offset = torch.tensor(scalers['metrics'].min_, dtype=torch.float32).to(config.DEVICE)

    total_spectra_mse = 0.0
    total_spectra_mae = 0.0
    total_metrics_mse = 0.0
    total_metrics_mae = 0.0
    num_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating Forward Model", leave=False)
        for batch in progress_bar:
            struct_params = batch['struct'].to(config.DEVICE)
            target_spectra = batch['spectra'].to(config.DEVICE)
            target_metrics = batch['metrics'].to(config.DEVICE)

            predicted_spectra, predicted_metrics = model(struct_params)

            # Spectra Loss
            total_spectra_mse += spectra_mse_criterion(predicted_spectra, target_spectra).item()
            total_spectra_mae += spectra_mae_criterion(predicted_spectra, target_spectra).item()

            # Metrics Loss (handle NaNs if any)
            predicted_metrics_cleaned = torch.nan_to_num(predicted_metrics, nan=0.0)
            # Normalize predicted metrics to the same space as targets: scaled = x * scale + offset
            predicted_metrics_scaled = predicted_metrics_cleaned * metrics_scale + metrics_offset
            target_metrics_cleaned = torch.nan_to_num(target_metrics, nan=0.0)
            total_metrics_mse += metrics_mse_criterion(predicted_metrics_scaled, target_metrics_cleaned).item()
            total_metrics_mae += metrics_mae_criterion(predicted_metrics_scaled, target_metrics_cleaned).item()

            num_samples += struct_params.shape[0]

    avg_spectra_mse = total_spectra_mse / num_samples
    avg_spectra_mae = total_spectra_mae / num_samples
    avg_metrics_mse = total_metrics_mse / num_samples
    avg_metrics_mae = total_metrics_mae / num_samples

    print("\n--- Forward Model Performance on Test Set ---")
    print(f"Spectra Prediction MSE: {avg_spectra_mse:.6f}")
    print(f"Spectra Prediction MAE: {avg_spectra_mae:.6f}")
    print(f"Metrics Prediction MSE: {avg_metrics_mse:.6f}")
    print(f"Metrics Prediction MAE: {avg_metrics_mae:.6f}")

    print("\n--- Performance Targets ---")
    print(f"Spectra Target MSE: < 0.045")
    print(f"Spectra Target MAE: < 2.0")
    print(f"Metrics Target MSE: < 0.01")
    print(f"Metrics Target MAE: < 0.1")

    print("Standalone forward model evaluation complete.")

if __name__ == '__main__':
    evaluate_forward_model_standalone()

import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.utils.data_loader import get_dataloaders
from core.models.forward_model import ForwardModel

def evaluate_forward_model(n_examples=4):
    """Evaluates the forward model on the test set for both spectra and metrics with visualization."""
    print("--- Starting Forward Model Comprehensive Evaluation ---")

    _, _, test_loader, scalers = get_dataloaders(batch_size=config.BATCH_SIZE)

    model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS),
        metrics_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)

    model_path = os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Forward model loaded from {model_path}\n")

    all_target_spectra = []
    all_predicted_spectra = []
    all_target_metrics = []
    all_predicted_metrics = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Forward Model"):
            struct_params = batch['struct'].to(config.DEVICE)
            target_spectra = batch['spectra'].to(config.DEVICE)
            target_metrics = batch['metrics'].to(config.DEVICE)

            predicted_spectra, predicted_metrics = model(struct_params)

            all_target_spectra.append(target_spectra.cpu().numpy())
            all_predicted_spectra.append(predicted_spectra.cpu().numpy())
            all_target_metrics.append(target_metrics.cpu().numpy())
            all_predicted_metrics.append(predicted_metrics.cpu().numpy())

    # Concatenate all batches
    all_target_spectra = np.concatenate(all_target_spectra, axis=0)
    all_predicted_spectra = np.concatenate(all_predicted_spectra, axis=0)
    all_target_metrics = np.concatenate(all_target_metrics, axis=0)
    all_predicted_metrics = np.concatenate(all_predicted_metrics, axis=0)

    # --- Calculate Metrics ---
    # Spectra
    spectra_mse = np.mean((all_target_spectra - all_predicted_spectra)**2)
    spectra_mae = np.mean(np.abs(all_target_spectra - all_predicted_spectra))
    spectra_r2 = r2_score(all_target_spectra, all_predicted_spectra)

    # Metrics
    metrics_mse = np.mean((all_target_metrics - all_predicted_metrics)**2)
    metrics_mae = np.mean(np.abs(all_target_metrics - all_predicted_metrics))
    metrics_r2 = r2_score(all_target_metrics, all_predicted_metrics)

    print("\n--- Forward Model Performance on Test Set ---")
    print("\n[Spectra Prediction]")
    print(f"  - MSE: {spectra_mse:.6f}")
    print(f"  - MAE: {spectra_mae:.6f}")
    print(f"  - R² Score: {spectra_r2:.4f}")
    
    print("\n[Metrics Prediction]")
    print(f"  - MSE: {metrics_mse:.6f}")
    print(f"  - MAE: {metrics_mae:.6f}")
    print(f"  - R² Score: {metrics_r2:.4f}")

    # --- Visualization ---
    # Inverse transform for plotting
    plot_target_spectra = scalers['spectra'].inverse_transform(all_target_spectra[:n_examples])
    plot_predicted_spectra = scalers['spectra'].inverse_transform(all_predicted_spectra[:n_examples])
    plot_target_metrics = scalers['metrics'].inverse_transform(all_target_metrics)
    plot_predicted_metrics = scalers['metrics'].inverse_transform(all_predicted_metrics)

    freq_axis = np.array([float(col.split('_')[1]) for col in config.SPECTRA_PARAMS])

    fig = plt.figure(figsize=(20, 10 + 4 * int(np.ceil(len(config.METRIC_PARAMS)/4))))
    gs = fig.add_gridspec(2 + int(np.ceil(len(config.METRIC_PARAMS)/4)), 4)
    fig.suptitle('Forward Model Evaluation', fontsize=20)

    # Spectra Plots
    for i in range(n_examples):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(freq_axis, plot_target_spectra[i], 'b-', label='Real')
        ax.plot(freq_axis, plot_predicted_spectra[i], 'r--', label='Predicted')
        ax.set_title(f'Sample Spectrum {i+1}')
        ax.grid(True, linestyle='--')
        ax.legend()

    # Metrics Scatter Plots
    for i, metric in enumerate(config.METRIC_PARAMS):
        row = 1 + i // 4
        col = i % 4
        ax = fig.add_subplot(gs[row, col])
        ax.scatter(plot_target_metrics[:, i], plot_predicted_metrics[:, i], alpha=0.5, s=10)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0)
        ax.set_xlabel("Real Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f'Metric: {metric}')
        ax.grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(config.PLOT_DIR, 'forward_model_evaluation.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"\nVisualization plot saved to {plot_path}")
    print("\nEvaluation complete.")

if __name__ == '__main__':
    evaluate_forward_model()
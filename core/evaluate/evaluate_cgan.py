
import torch
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.models.generator import Generator
from core.models.forward_model import ForwardModel
from core.utils.data_loader import get_dataloaders

def evaluate_cgan(n_examples=5):
    """Evaluates the conditional GAN's ability to perform inverse design."""
    print("Starting Conditional GAN (cGAN) evaluation...")

    # --- Load Models ---
    # Load Generator
    generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS),
        condition_dim=config.CONDITION_DIM
    ).to(config.DEVICE)
    gen_path = os.path.join(config.SAVED_MODELS_DIR, 'final_cgan_generator.pth')
    if not os.path.exists(gen_path):
        print(f"Generator model not found at {gen_path}")
        return
    generator.load_state_dict(torch.load(gen_path))
    generator.eval()
    print(f"Conditional generator loaded from {gen_path}")

    # Load Forward Model
    forward_model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS),
        metrics_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)
    fm_path = os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')
    if not os.path.exists(fm_path):
        print(f"Forward model not found at {fm_path}")
        return
    forward_model.load_state_dict(torch.load(fm_path))
    forward_model.eval()
    print(f"Forward model loaded from {fm_path}")

    # --- Evaluation ---
    _, _, test_loader, scalers = get_dataloaders(batch_size=1000) # Use a larger sample for robust stats
    test_batch = next(iter(test_loader))

    real_spectra_norm = test_batch['spectra'].to(config.DEVICE)
    condition_norm = test_batch['metrics'].to(config.DEVICE)

    # Generate structures based on the condition (metrics)
    z = torch.randn(condition_norm.size(0), config.LATENT_DIM).to(config.DEVICE)
    with torch.no_grad():
        fake_structs_norm = generator(z, condition_norm)
        # Predict spectra and metrics for the generated structures
        fake_spectra_norm, fake_metrics_norm = forward_model(fake_structs_norm)

    # --- Quantitative Evaluation ---
    print("\n--- Quantitative Evaluation ---")
    
    # 1. Spectral Fidelity
    spectral_mse = torch.nn.functional.mse_loss(fake_spectra_norm, real_spectra_norm)
    print(f"1. Spectral Fidelity (MSE between generated and target spectra): {spectral_mse.item():.6f}")

    # 2. Metric Fidelity
    metric_mse = torch.nn.functional.mse_loss(fake_metrics_norm, condition_norm)
    print(f"2. Metric Fidelity (MSE between generated and target metrics): {metric_mse.item():.6f}")

    # 3. Physical Constraint Adherence
    print("\n--- Physical Constraint Adherence ---")
    fake_structs = scalers['struct'].inverse_transform(fake_structs_norm.cpu().numpy())
    fake_structs_df = pd.DataFrame(fake_structs, columns=config.STRUCT_PARAMS)
    if 'r1' in fake_structs_df.columns and 'r2' in fake_structs_df.columns:
        valid_constraints = fake_structs_df['r1'] >= fake_structs_df['r2']
        adherence_percentage = valid_constraints.mean() * 100
        print(f"Adherence to 'r1 >= r2' constraint: {adherence_percentage:.2f}%")
    else:
        print("Skipping 'r1 >= r2' constraint check as columns not found.")

    # --- Plotting ---
    # Inverse transform for plotting
    real_spectra = scalers['spectra'].inverse_transform(real_spectra_norm.cpu().numpy()[:n_examples])
    fake_spectra = scalers['spectra'].inverse_transform(fake_spectra_norm.cpu().numpy()[:n_examples])
    
    freq_axis = np.array([float(col.split('_')[1]) for col in config.SPECTRA_PARAMS])
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(10, 4 * n_examples), sharex=True)
    fig.suptitle('cGAN Inverse Design Evaluation', fontsize=16)

    for i in range(n_examples):
        ax = axes[i]
        ax.plot(freq_axis, real_spectra[i], color='blue', label=f'Target Spectrum {i+1}')
        ax.plot(freq_axis, fake_spectra[i], color='red', linestyle='--', label=f'Generated Spectrum {i+1}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_ylabel('Transmission')
    
    axes[-1].set_xlabel('Frequency (THz)')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    plot_path = os.path.join(config.PLOT_DIR, 'cgan_inverse_design_evaluation.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nQualitative evaluation plot saved to {plot_path}")

if __name__ == '__main__':
    evaluate_cgan()

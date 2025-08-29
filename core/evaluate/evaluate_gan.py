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

def gaussian_kernel(x, y, sigma=1.0):
    """Gaussian Kernel for MMD calculation."""
    beta = 1. / (2. * sigma**2)
    dist = torch.cdist(x, y, p=2).pow(2)
    return torch.exp(-beta * dist)

def mmd_loss(x, y, sigma=1.0):
    """Maximum Mean Discrepancy loss."""
    xx = gaussian_kernel(x, x, sigma).mean()
    yy = gaussian_kernel(y, y, sigma).mean()
    xy = gaussian_kernel(x, y, sigma).mean()
    return xx + yy - 2 * xy

def evaluate_gan(n_samples=1000):
    """Performs a detailed quantitative and qualitative evaluation of the GAN."""
    print("Starting detailed GAN evaluation with MMD...")

    # --- Load Models ---
    generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS),
        condition_dim=0
    ).to(config.DEVICE)
    gen_path = os.path.join(config.SAVED_MODELS_DIR, 'generator_epoch_1000.pth')
    generator.load_state_dict(torch.load(gen_path))
    generator.eval()
    print(f"Generator loaded from {gen_path}")

    forward_model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS),
        metrics_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)
    fm_path = os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')
    forward_model.load_state_dict(torch.load(fm_path))
    forward_model.eval()
    print(f"Forward model loaded from {fm_path}")

    # --- Generate Fake Data ---
    z = torch.randn(n_samples, config.LATENT_DIM).to(config.DEVICE)
    with torch.no_grad():
        fake_structs_norm_tensor = generator(z, None)
        fake_spectra_norm_tensor, _ = forward_model(fake_structs_norm_tensor)

    # --- Load Real Data ---
    _, _, test_loader, scalers = get_dataloaders(batch_size=n_samples)
    real_structs_norm_list = []
    real_spectra_norm_list = []
    for batch in test_loader:
        real_structs_norm_list.append(batch['struct'])
        real_spectra_norm_list.append(batch['spectra'])
    
    real_structs_norm_tensor = torch.cat(real_structs_norm_list).to(config.DEVICE)
    real_spectra_norm_tensor = torch.cat(real_spectra_norm_list).to(config.DEVICE)

    # --- 1. MMD & Statistical Analysis of Structural Parameters (Normalized Space) ---
    print("\n--- 1. Analysis of Structural Parameters ---")
    mmd_struct = mmd_loss(real_structs_norm_tensor, fake_structs_norm_tensor, sigma=1.0)
    print(f"MMD between real and generated structural parameters: {mmd_struct.item():.6f}")

    real_structs = scalers['struct'].inverse_transform(real_structs_norm_tensor.cpu().numpy())
    fake_structs = scalers['struct'].inverse_transform(fake_structs_norm_tensor.cpu().numpy())
    real_structs_df = pd.DataFrame(real_structs, columns=config.STRUCT_PARAMS)
    fake_structs_df = pd.DataFrame(fake_structs, columns=config.STRUCT_PARAMS)
    print("\n[REAL DATA STATS]")
    print(real_structs_df.describe())
    print("\n[GENERATED DATA STATS]")
    print(fake_structs_df.describe())

    # --- 2. MMD & Statistical Analysis of Spectra (Normalized Space) ---
    print("\n--- 2. Analysis of Spectra Distributions ---")
    mmd_spectra = mmd_loss(real_spectra_norm_tensor, fake_spectra_norm_tensor, sigma=1.0)
    print(f"MMD between real and generated spectra: {mmd_spectra.item():.6f}")

    # --- 3. Physical Constraint Check ---
    print("\n--- 3. Physical Constraint Adherence ---")
    if 'r1' in fake_structs_df.columns and 'r2' in fake_structs_df.columns:
        valid_constraints = fake_structs_df['r1'] >= fake_structs_df['r2']
        adherence_percentage = valid_constraints.mean() * 100
        print(f"Adherence to 'r1 >= r2' constraint: {adherence_percentage:.2f}%")
    else:
        print("Skipping 'r1 >= r2' constraint check as columns not found.")

    # Inverse transform for plotting
    real_spectra_plot = scalers['spectra'].inverse_transform(real_spectra_norm_tensor.cpu().numpy())
    fake_spectra_plot = scalers['spectra'].inverse_transform(fake_spectra_norm_tensor.cpu().numpy())

    # --- 4. Plotting ---
    mean_real_spectra = np.mean(real_spectra_plot, axis=0)
    std_real_spectra = np.std(real_spectra_plot, axis=0)
    mean_fake_spectra = np.mean(fake_spectra_plot, axis=0)
    std_fake_spectra = np.std(fake_spectra_plot, axis=0)
    freq_axis = np.array([float(col.split('_')[1]) for col in config.SPECTRA_PARAMS])

    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Detailed GAN Evaluation with MMD', fontsize=16)

    # a) Sample Spectra
    for i in range(5):
        axs[0, 0].plot(freq_axis, real_spectra_plot[i], color='blue', alpha=0.6, label='Real' if i == 0 else "")
        axs[0, 0].plot(freq_axis, fake_spectra_plot[i], color='red', linestyle='--', alpha=0.8, label='Generated' if i == 0 else "")
    axs[0, 0].set_title('Sample Spectra: Real (blue) vs. Generated (red)')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--')

    # b) Mean Spectra
    axs[0, 1].plot(freq_axis, mean_real_spectra, color='blue', label='Mean Real Spectra')
    axs[0, 1].fill_between(freq_axis, mean_real_spectra - std_real_spectra, mean_real_spectra + std_real_spectra, color='blue', alpha=0.2)
    axs[0, 1].plot(freq_axis, mean_fake_spectra, color='red', linestyle='--', label='Mean Generated Spectra')
    axs[0, 1].fill_between(freq_axis, mean_fake_spectra - std_fake_spectra, mean_fake_spectra + std_fake_spectra, color='red', alpha=0.2)
    axs[0, 1].set_title('Mean Spectra and Standard Deviation')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--')

    # c) & d) Parameter Distributions
    for i, param in enumerate(config.STRUCT_PARAMS[:2]):
        axs[1, i].hist(real_structs_df[param], bins=30, alpha=0.7, label='Real', color='blue', density=True)
        axs[1, i].hist(fake_structs_df[param], bins=30, alpha=0.7, label='Generated', color='red', density=True)
        axs[1, i].set_title(f'Distribution of Parameter: {param}')
        axs[1, i].legend()
        axs[1, i].grid(True, linestyle='--')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(config.PLOT_DIR, 'gan_detailed_evaluation.png')
    plt.savefig(plot_path)
    plt.close()

    print(f"\nDetailed evaluation complete. Plot saved to {plot_path}")

if __name__ == '__main__':
    evaluate_gan()
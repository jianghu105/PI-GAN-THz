# PI_GAN_THZ/core/utils/plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os

# Removed setup_matplotlib_for_chinese as per request to remove Chinese characters.

def plot_losses(epochs: list, losses: dict, title: str, xlabel: str, ylabel: str, save_path: str):
    """
    Plots multiple loss curves.

    Args:
        epochs (list): List of epochs corresponding to the loss values.
        losses (dict): Dictionary containing multiple loss curves.
                       Keys are curve names (labels), and values are lists of loss values.
                       Example: {'Loss A': [val1, val2, ...], 'Loss B': [val1, val2, ...]}
        title (str): Title of the plot.
        xlabel (str): Label for the X-axis.
        ylabel (str): Label for the Y-axis.
        save_path (str): Full path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    for label, data in losses.items():
        plt.plot(epochs, data, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel) # Using the provided xlabel
    plt.ylabel(ylabel) # Using the provided ylabel
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")

def plot_generated_samples(
    real_spectrums: np.ndarray, 
    recon_spectrums: np.ndarray, 
    real_params: np.ndarray, 
    predicted_params: np.ndarray, 
    frequencies: np.ndarray,
    num_samples: int = 5, 
    save_path: str = 'generated_samples.png'
):
    """
    Visualizes real vs. reconstructed spectra and their corresponding real vs. predicted parameters
    (for PI-GAN generation results).

    Args:
        real_spectrums (np.ndarray): Real spectrum data (batch_size, spectrum_dim).
        recon_spectrums (np.ndarray): Reconstructed spectrum data (batch_size, spectrum_dim).
        real_params (np.ndarray): Real structural parameters (batch_size, param_dim).
        predicted_params (np.ndarray): Predicted structural parameters (batch_size, param_dim).
        frequencies (np.ndarray): Frequency points corresponding to the spectrum (spectrum_dim,).
        num_samples (int): Number of samples to visualize.
        save_path (str): Path to save the plot.
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, num_samples * 4)) # Two columns: spectrum comparison and parameter comparison

    if num_samples == 1: # If only one sample, axes might not be a 2D array, handle it
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # First column: Spectrum Comparison
        ax1 = axes[i, 0]
        ax1.plot(frequencies, real_spectrums[i], label='Real Spectrum', color='blue')
        ax1.plot(frequencies, recon_spectrums[i], label='Reconstructed Spectrum', color='red', linestyle='--')
        ax1.set_title(f'Sample {i+1}: Spectrum Comparison')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True)

        # Second column: Parameter Comparison
        ax2 = axes[i, 1]
        param_indices = np.arange(real_params.shape[1])
        ax2.bar(param_indices - 0.2, real_params[i], width=0.4, label='Real Parameters', color='skyblue')
        ax2.bar(param_indices + 0.2, predicted_params[i], width=0.4, label='Predicted Parameters', color='lightcoral')
        ax2.set_xticks(param_indices)
        ax2.set_xticklabels([f'P{j+1}' for j in param_indices]) 
        ax2.set_title(f'Sample {i+1}: Parameter Comparison')
        ax2.set_xlabel('Parameter Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated sample plots saved to: {save_path}")

def plot_fwd_model_predictions(
    real_params: np.ndarray,
    real_spectrums: np.ndarray,
    predicted_spectrums: np.ndarray,
    real_metrics: np.ndarray,
    predicted_metrics: np.ndarray,
    frequencies: np.ndarray,
    num_samples: int = 5,
    save_path: str = 'fwd_model_predictions.png',
    metric_names: list = None
):
    """
    Visualizes forward model predictions for Spectra and Metrics.

    Args:
        real_params (np.ndarray): Real structural parameters (batch_size, param_dim).
        real_spectrums (np.ndarray): Real spectrum data (batch_size, spectrum_dim).
        predicted_spectrums (np.ndarray): Predicted spectrum data (batch_size, spectrum_dim).
        real_metrics (np.ndarray): Real physical metric data (batch_size, metrics_dim).
        predicted_metrics (np.ndarray): Predicted physical metric data (batch_size, metrics_dim).
        frequencies (np.ndarray): Frequency points corresponding to the spectrum (spectrum_dim,).
        num_samples (int): Number of samples to visualize.
        save_path (str): Path to save the plot.
        metric_names (list): List of names for physical metrics, used for legends.
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, num_samples * 4)) # Three columns: Parameters, Spectrum Comparison, Metrics Comparison

    if num_samples == 1: # If only one sample, axes might not be a 2D array, handle it
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # First column: Input Parameters
        ax1 = axes[i, 0]
        param_indices = np.arange(real_params.shape[1])
        ax1.bar(param_indices, real_params[i], color='lightgray')
        ax1.set_xticks(param_indices)
        ax1.set_xticklabels([f'P{j+1}' for j in param_indices])
        ax1.set_title(f'Sample {i+1}: Input Parameters')
        ax1.set_ylabel('Value')
        ax1.grid(True, axis='y')

        # Second column: Spectrum Comparison
        ax2 = axes[i, 1]
        ax2.plot(frequencies, real_spectrums[i], label='Real Spectrum', color='blue')
        ax2.plot(frequencies, predicted_spectrums[i], label='Predicted Spectrum', color='red', linestyle='--')
        ax2.set_title(f'Sample {i+1}: Spectrum Prediction')
        ax2.set_xlabel('Frequency (THz)')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True)

        # Third column: Metrics Comparison
        ax3 = axes[i, 2]
        metric_indices = np.arange(real_metrics.shape[1])
        width = 0.35
        ax3.bar(metric_indices - width/2, real_metrics[i], width, label='Real Metrics', color='skyblue')
        ax3.bar(metric_indices + width/2, predicted_metrics[i], width, label='Predicted Metrics', color='lightcoral')
        ax3.set_xticks(metric_indices)
        ax3.set_xticklabels(metric_names if metric_names else [f'M{j+1}' for j in metric_indices], rotation=45, ha='right')
        ax3.set_title(f'Sample {i+1}: Metrics Prediction')
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plots saved to: {save_path}")
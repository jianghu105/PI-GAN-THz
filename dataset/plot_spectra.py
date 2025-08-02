
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import config

def plot_all_spectra(output_dir="./plots", num_samples_to_plot=4450):
    """Loads the dataset and plots all transmission spectra, highlighting resonance features.

    Args:
        output_dir (str): Directory to save the plots.
        num_samples_to_plot (int): Number of spectra to plot. Set to 4450 for all.
    """
    # Ensure the dataset path is correct relative to the project root
    project_root = os.getcwd() # Use current working directory as project root
    dataset_full_path = os.path.join(project_root, config.DATASET_PATH)

    print(f"Loading data from {dataset_full_path}...")
    print(f"DEBUG: dataset_full_path = {dataset_full_path}")
    try:
        df = pd.read_csv(dataset_full_path)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_full_path}. Please check the path.")
        return

    os.makedirs(output_dir, exist_ok=True)

    spectra_cols = config.SPECTRA_PARAMS
    freq_values = np.array([float(col.split('_')[1]) for col in spectra_cols])

    print(f"Plotting {min(num_samples_to_plot, len(df))} spectra...")

    plt.figure(figsize=(16, 10))
    ax = plt.gca()

    # Plot all spectra with transparency
    for i in range(min(num_samples_to_plot, len(df))):
        plt.plot(freq_values, df.iloc[i][spectra_cols], color='#D3D3D3', alpha=0.05) # Light gray, very transparent

    # Plot a few random spectra with higher opacity to show individual curves
    random_indices = np.random.choice(len(df), min(50, len(df)), replace=False)
    for i in random_indices:
        plt.plot(freq_values, df.iloc[i][spectra_cols], color='#1f77b4', alpha=0.3, linewidth=1) # Matplotlib default blue

    # Overlay average spectrum
    mean_spectrum = df[spectra_cols].mean()
    plt.plot(freq_values, mean_spectrum, color='#ff7f0e', linewidth=2, label='Average Spectrum') # Matplotlib default orange

    # Plot resonance peak positions (f1, f2) as scatter points
    # Use seaborn kdeplot to visualize density of peak positions
    sns.kdeplot(x=df['f1'], y=df['f2'], cmap="plasma", fill=True, ax=ax, alpha=0.6, label='Peak Frequencies Density')

    # Add scatter points for Q1 and Q2 (as color or size)
    # For simplicity, let's just plot f1 and f2, and indicate Q1/Q2 with color/size if possible
    # This might be too cluttered with 4450 points, so we'll use density plots for f1/f2
    # and perhaps a separate plot for Q/FoM distributions.

    # Add labels and title
    plt.title('THz Metamaterial Transmission Spectra Overview', fontsize=18)
    plt.xlabel('Frequency (THz)', fontsize=14)
    plt.ylabel('Magnitude (dB)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Add text annotations for general observations about peaks and bandwidth
    plt.text(0.02, 0.98, 'Observation: Two main resonance peaks are visible.', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.02, 0.95, 'Peak positions (f1, f2) vary across samples.', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    plt.text(0.02, 0.92, 'Bandwidth (related to Q-factor) also shows variation.', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plot_filename = os.path.join(output_dir, 'all_spectra_overview.png')
    plt.savefig(plot_filename)
    plt.close()
    print(f"Plot saved to {plot_filename}")

    # Separate plots for Q-factor and FoM distributions for clarity
    plt.figure(figsize=(12, 6))
    sns.histplot(df['Q1'], color='purple', kde=True, label='Q1', stat='density', alpha=0.6)
    sns.histplot(df['Q2'], color='orange', kde=True, label='Q2', stat='density', alpha=0.6)
    plt.title('Distribution of Q-factors', fontsize=16)
    plt.xlabel('Q-factor', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    q_plot_filename = os.path.join(output_dir, 'q_factor_distribution.png')
    plt.savefig(q_plot_filename)
    plt.close()
    print(f"Q-factor distribution plot saved to {q_plot_filename}")

    plt.figure(figsize=(12, 6))
    sns.histplot(df['FoM1'], color='green', kde=True, label='FoM1', stat='density', alpha=0.6)
    sns.histplot(df['FoM2'], color='brown', kde=True, label='FoM2', stat='density', alpha=0.6)
    plt.title('Distribution of FoM', fontsize=16)
    plt.xlabel('FoM', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    fom_plot_filename = os.path.join(output_dir, 'fom_distribution.png')
    plt.savefig(fom_plot_filename)
    plt.close()
    print(f"FoM distribution plot saved to {fom_plot_filename}")


if __name__ == '__main__':
    # Ensure the dataset path is correct relative to the project root
    project_root = os.getcwd() # Use current working directory as project root
    plot_all_spectra(output_dir=os.path.join(project_root, "plots"))

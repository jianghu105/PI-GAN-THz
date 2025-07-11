import sys
import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader 
from tqdm.notebook import tqdm # Import tqdm.notebook for Colab compatibility

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import models, data loader, and plotting utilities
from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.plot_utils import plot_losses, plot_generated_samples
from core.utils.set_seed import set_seed

def evaluate_pigan(num_samples_to_plot: int = 5):
    """
    Evaluates and visualizes the training results of the PI-GAN.

    Args:
        num_samples_to_plot (int): Number of generated samples to visualize.
    """
    tqdm.write("\n--- Starting PI-GAN Evaluation Script ---") 

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"Using device: {device}") 

    # Set random seed for reproducibility
    set_seed(cfg.RANDOM_SEED)
    tqdm.write(f"Random seed set to: {cfg.RANDOM_SEED}") 

    # Ensure necessary directories are created, including plots directory
    cfg.create_directories() 
    tqdm.write(f"Evaluation plots will be saved to: {cfg.PLOTS_DIR}") 

    # --- Data Loading ---
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        tqdm.write(f"Error: Dataset not found at {data_path}. Please check config.py and ensure the CSV file exists.") 
        return 
    
    # Note: Evaluating PI-GAN usually involves sampling directly from the dataset, 
    # rather than iterating through a DataLoader in batches for the entire dataset.
    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    tqdm.write(f"Dataset size: {len(dataset)} samples") 

    # --- Model Initialization and Loading ---
    generator = Generator(input_dim=cfg.SPECTRUM_DIM, output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM).to(device)
    discriminator = Discriminator(input_spec_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM, 
                                  input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM).to(device)
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    # Construct model file paths
    gen_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth")
    disc_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth")
    # Assuming the forward model is also saved as a final version after PI-GAN training.
    # If not, use the pre-trained version path (e.g., "forward_model_pretrained.pth")
    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth") 

    # Check if all model files exist
    models_found = True
    if not os.path.exists(gen_model_path):
        tqdm.write(f"Error: Generator model not found at {gen_model_path}.") 
        models_found = False
    if not os.path.exists(disc_model_path):
        tqdm.write(f"Error: Discriminator model not found at {disc_model_path}.") 
        models_found = False
    if not os.path.exists(fwd_model_path):
        tqdm.write(f"Error: ForwardModel model not found at {fwd_model_path}.") 
        models_found = False
    
    if not models_found:
        tqdm.write("Please ensure PI-GAN training is complete and all models are saved, or specify correct paths.") 
        return # Gracefully return if any model is missing

    try:
        generator.load_state_dict(torch.load(gen_model_path, map_location=device))
        discriminator.load_state_dict(torch.load(disc_model_path, map_location=device))
        forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device)) # Load forward model weights
    except Exception as e:
        tqdm.write(f"Error: An exception occurred while loading models: {e}") 
        tqdm.write("Please check if model files are corrupted or do not match the model architecture.") 
        return

    generator.eval() # Switch to evaluation mode
    discriminator.eval() # Switch to evaluation mode
    forward_model.eval() # Switch to evaluation mode
    tqdm.write(f"Final Generator, Discriminator, and ForwardModel loaded from {cfg.SAVED_MODELS_DIR}.") 

    # --- Load Loss History ---
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "pigan_loss_history.pt")
    if not os.path.exists(loss_history_path):
        tqdm.write(f"Warning: PI-GAN training loss history not found at {loss_history_path}. Cannot generate loss plots.") 
        loss_history = {}
    else:
        try:
            # Assuming pigan_loss_history.pt saves a dictionary containing various loss lists
            loaded_history = torch.load(loss_history_path)
            # Check for required keys
            required_keys = ['g_losses', 'd_losses', 'adv_losses', 'recon_spec_losses', 
                             'recon_metrics_losses', 'maxwell_losses', 'lc_losses', 
                             'param_range_losses', 'bnn_kl_losses']
            
            if isinstance(loaded_history, dict) and all(k in loaded_history for k in required_keys):
                loss_history = loaded_history
                tqdm.write(f"PI-GAN training loss history loaded from {loss_history_path}.") 
            else:
                tqdm.write(f"Warning: Loss history file {loss_history_path} has incorrect format or missing key data. Cannot generate loss plots.") 
                loss_history = {}
        except Exception as e:
            tqdm.write(f"Error: An exception occurred while loading loss history: {e}. Cannot generate loss plots.") 
            loss_history = {}

    # --- Plot Loss Curves ---
    # Ensure each loss list is not empty and epoch_for_plot length matches loss list
    if loss_history and 'g_losses' in loss_history and loss_history['g_losses']:
        # Assuming loss_history lists have consistent length
        epochs_for_plot = list(range(1, len(loss_history['g_losses']) + 1)) 
        
        tqdm.write("\n--- Generating PI-GAN Training Loss Plots ---") 
        # Plot total losses
        plot_losses(
            epochs=epochs_for_plot,
            losses={
                'Generator Loss': loss_history.get('g_losses', []), # Translated to English
                'Discriminator Loss': loss_history.get('d_losses', []) # Translated to English
            },
            title='Generator and Discriminator Losses over Epochs', # Translated to English
            xlabel='Epoch',
            ylabel='Loss', # Translated to English
            save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_gan_losses.png')
        )
        # Image of PI-GAN GAN losses
        # Plot generator sub-losses
        plot_losses(
            epochs=epochs_for_plot,
            losses={
                'Adversarial Loss': loss_history.get('adv_losses', []), # Translated to English
                'Spectrum Reconstruction Loss': loss_history.get('recon_spec_losses', []), # Translated to English
                'Metrics Reconstruction Loss': loss_history.get('recon_metrics_losses', []), # Translated to English
                'Maxwell Loss': loss_history.get('maxwell_losses', []), # Translated to English
                'LC Model Loss': loss_history.get('lc_losses', []), # Translated to English
                'Parameter Range Loss': loss_history.get('param_range_losses', []), # Translated to English
                'BNN KL Divergence Loss': loss_history.get('bnn_kl_losses', []) # Translated to English
            },
            title='Generator Sub-Losses over Epochs', # Translated to English
            xlabel='Epoch',
            ylabel='Loss', # Translated to English
            save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_generator_sub_losses.png')
        )
        # Image of PI-GAN generator sub-losses
        tqdm.write(f"Loss plots saved to {cfg.PLOTS_DIR}") 
    else:
        tqdm.write("No valid loss history data available for plotting.") 

    # --- Generate Sample Visualization ---
    tqdm.write("\n--- Generating PI-GAN Sample Visualization ---") 
    with torch.no_grad():
        if num_samples_to_plot <= 0:
            tqdm.write("Warning: Number of samples to plot is 0 or negative, skipping sample visualization.") 
            
        elif num_samples_to_plot > len(dataset):
            tqdm.write(f"Warning: Number of samples to plot ({num_samples_to_plot}) exceeds dataset size ({len(dataset)}). Plotting all {len(dataset)} samples.") 
            num_samples_to_plot = len(dataset)
        
        if num_samples_to_plot == 0: # Re-check if adjusted to 0
            tqdm.write("Not enough samples available for generating visualization.") 
            
        else:
            # Randomly select a batch of real spectra for generation and visualization
            sample_indices = np.random.choice(len(dataset), num_samples_to_plot, replace=False) 
            
            # Use DataLoader to efficiently get these samples, even for a single batch
            from torch.utils.data import Subset
            subset_dataset = Subset(dataset, sample_indices)
            sample_dataloader = DataLoader(subset_dataset, batch_size=num_samples_to_plot, shuffle=False)
            
            # Get the first (and only) batch from the dataloader
            sample_batch = next(iter(sample_dataloader)) 
            
            sample_real_spectrums = sample_batch[0].to(device)
            sample_real_params_denorm = sample_batch[1].to(device) # Real unnormalized parameters

            # Predict parameters via the Generator
            sample_predicted_params_norm = generator(sample_real_spectrums)
            sample_predicted_params_denorm = denormalize_params(sample_predicted_params_norm, dataset.param_ranges)

            # Reconstruct spectra via the Forward Model (for cycle consistency check)
            recon_spectrums, _ = forward_model(sample_predicted_params_norm)

            # Move tensors back to CPU and convert to NumPy for plotting
            sample_real_spectrums_np = sample_real_spectrums.cpu().numpy()
            recon_spectrums_np = recon_spectrums.cpu().numpy()
            sample_real_params_denorm_np = sample_real_params_denorm.cpu().numpy()
            sample_predicted_params_denorm_np = sample_predicted_params_denorm.cpu().numpy()
            
            frequencies = dataset.frequencies # Get frequencies from MetamaterialDataset

            plot_generated_samples(
                real_spectrums=sample_real_spectrums_np,
                recon_spectrums=recon_spectrums_np,
                real_params=sample_real_params_denorm_np,
                predicted_params=sample_predicted_params_denorm_np,
                frequencies=frequencies,
                num_samples=num_samples_to_plot,
                save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_generated_samples.png')
            )
            # Image of PI-GAN generated samples
            tqdm.write(f"PI-GAN sample plot saved to {cfg.PLOTS_DIR}") 
    
    tqdm.write("--- PI-GAN Evaluation Script Completed ---") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained PI-GAN model.")
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize (default: 5)')
    args = parser.parse_args()
    
    evaluate_pigan(num_samples_to_plot=args.num_samples)
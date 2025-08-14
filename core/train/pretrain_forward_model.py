
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt # Import matplotlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.utils.data_loader import get_dataloaders
from core.models.forward_model import ForwardModel

def pretrain_forward_model():
    """Pre-trains the forward model to predict spectra from structural parameters
    and extract metrics."""
    print("Starting forward model pre-training...")

    # Create directories if they don't exist
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True) # Ensure log directory exists for console logs
    os.makedirs(config.PLOT_DIR, exist_ok=True) # Ensure plot directory exists for plots

    # Get DataLoaders and scalers (for metric normalization alignment)
    train_loader, val_loader, _, scalers = get_dataloaders(batch_size=config.BATCH_SIZE)

    # Initialize model, optimizer, and loss function
    model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS)
    ).to(config.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=config.PRETRAIN_FWD_MODEL_LR)
    
    # Use MSELoss for both spectra and metrics
    spectra_criterion = nn.MSELoss()
    metrics_criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 20  # Number of epochs to wait for improvement before stopping

    # Lists to store loss history for plotting
    train_losses = []
    val_losses = []

    # Prepare scaler tensors for metrics normalization: scaled = x * scale + offset
    metrics_scale = torch.tensor(scalers['metrics'].scale_, dtype=torch.float32).to(config.DEVICE)
    metrics_offset = torch.tensor(scalers['metrics'].min_, dtype=torch.float32).to(config.DEVICE)

    for epoch in range(config.PRETRAIN_FWD_MODEL_EPOCHS):
        model.train()
        train_loss = 0.0
        train_spectra_loss = 0.0
        train_metrics_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.PRETRAIN_FWD_MODEL_EPOCHS}", leave=False)
        for batch in progress_bar:
            struct_params = batch['struct'].to(config.DEVICE)
            target_spectra = batch['spectra'].to(config.DEVICE)
            target_metrics = batch['metrics'].to(config.DEVICE) # Get target metrics

            optimizer.zero_grad()

            predicted_spectra, predicted_metrics = model(struct_params)

            # Calculate loss for spectra
            loss_spectra = spectra_criterion(predicted_spectra, target_spectra)
            
            # Calculate loss for metrics in the SAME normalized space as targets
            # Normalize predicted metrics with train-fitted scaler: scaled = x * scale + offset
            predicted_metrics_cleaned = torch.nan_to_num(predicted_metrics, nan=0.0)
            predicted_metrics_scaled = predicted_metrics_cleaned * metrics_scale + metrics_offset
            target_metrics_cleaned = torch.nan_to_num(target_metrics, nan=0.0)
            loss_metrics = metrics_criterion(predicted_metrics_scaled, target_metrics_cleaned)

            # Combine losses (you can add weights here if needed, e.g., 0.5 * loss_spectra + 0.5 * loss_metrics)
            loss = loss_spectra + loss_metrics # Simple sum for now

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_spectra_loss += loss_spectra.item()
            train_metrics_loss += loss_metrics.item()
            progress_bar.set_postfix(total_loss=loss.item(), spectra_loss=loss_spectra.item(), metrics_loss=loss_metrics.item())

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_spectra_loss = 0.0
        val_metrics_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                struct_params = batch['struct'].to(config.DEVICE)
                target_spectra = batch['spectra'].to(config.DEVICE)
                target_metrics = batch['metrics'].to(config.DEVICE)

                predicted_spectra, predicted_metrics = model(struct_params)
                
                loss_spectra = spectra_criterion(predicted_spectra, target_spectra)
                predicted_metrics_cleaned = torch.nan_to_num(predicted_metrics, nan=0.0)
                predicted_metrics_scaled = predicted_metrics_cleaned * metrics_scale + metrics_offset
                target_metrics_cleaned = torch.nan_to_num(target_metrics, nan=0.0)
                loss_metrics = metrics_criterion(predicted_metrics_scaled, target_metrics_cleaned)
                
                loss = loss_spectra + loss_metrics

                val_loss += loss.item()
                val_spectra_loss += loss_spectra.item()
                val_metrics_loss += loss_metrics.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_spectra_loss = train_spectra_loss / len(train_loader)
        avg_train_metrics_loss = train_metrics_loss / len(train_loader)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_spectra_loss = val_spectra_loss / len(val_loader)
        avg_val_metrics_loss = val_metrics_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f} (Spec: {avg_train_spectra_loss:.6f}, Met: {avg_train_metrics_loss:.6f}), "
              f"Val Loss: {avg_val_loss:.6f} (Spec: {avg_val_spectra_loss:.6f}, Met: {avg_val_metrics_loss:.6f})")

        # Store losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # Early stopping and model saving based on total validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth'))
            print(f"Validation loss improved. Saving best model to {config.SAVED_MODELS_DIR}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    # Save the final model
    torch.save(model.state_dict(), os.path.join(config.SAVED_MODELS_DIR, 'final_forward_model.pth'))
    print("Finished pre-training forward model.")

    # Plotting the loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Forward Model Pre-training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(config.PLOT_DIR, 'forward_model_pretrain_loss_curve.png') # Changed to PLOT_DIR
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to {plot_path}")

if __name__ == '__main__':
    pretrain_forward_model()

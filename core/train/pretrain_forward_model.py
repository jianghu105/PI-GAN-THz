
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.utils.data_loader import get_dataloaders
from core.models.forward_model import ForwardModel

def pretrain_forward_model():
    """Pre-trains the forward model to predict spectra from structural parameters."""
    print("Starting forward model pre-training...")

    # Create directories if they don't exist
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Get DataLoaders
    train_loader, val_loader, _, _ = get_dataloaders(batch_size=config.BATCH_SIZE)

    # Initialize model, optimizer, and loss function
    model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS)
    ).to(config.DEVICE)

    optimizer = optim.Adam(model.network.parameters(), lr=config.PRETRAIN_FWD_MODEL_LR)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 20  # Number of epochs to wait for improvement before stopping

    for epoch in range(config.PRETRAIN_FWD_MODEL_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.PRETRAIN_FWD_MODEL_EPOCHS}", leave=False)
        for batch in progress_bar:
            struct_params = batch['struct'].to(config.DEVICE)
            target_spectra = batch['spectra'].to(config.DEVICE)

            optimizer.zero_grad()

            # We only care about the predicted spectra for pre-training
            predicted_spectra, _ = model(struct_params)

            loss = criterion(predicted_spectra, target_spectra)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                struct_params = batch['struct'].to(config.DEVICE)
                target_spectra = batch['spectra'].to(config.DEVICE)

                predicted_spectra, _ = model(struct_params)
                loss = criterion(predicted_spectra, target_spectra)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # Early stopping and model saving
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

if __name__ == '__main__':
    pretrain_forward_model()

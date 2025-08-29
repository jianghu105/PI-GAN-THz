import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.utils.data_loader import get_dataloaders
from core.models.forward_model import ForwardModel

def pretrain_forward_model():
    """Pre-trains the simplified forward model to predict spectra from structural parameters."""
    print("Starting simplified forward model pre-training...")

    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)

    train_loader, val_loader, _, _ = get_dataloaders(batch_size=config.BATCH_SIZE)

    model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS),
        metrics_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=config.PRETRAIN_FWD_MODEL_LR, weight_decay=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 20

    train_losses = []
    val_losses = []

    for epoch in range(config.PRETRAIN_FWD_MODEL_EPOCHS):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.PRETRAIN_FWD_MODEL_EPOCHS}", leave=False)
        for batch in progress_bar:
            struct_params = batch['struct'].to(config.DEVICE)
            target_spectra = batch['spectra'].to(config.DEVICE)
            target_metrics = batch['metrics'].to(config.DEVICE)

            optimizer.zero_grad()

            predicted_spectra, predicted_metrics = model(struct_params)
            
            loss_spectra = criterion(predicted_spectra, target_spectra)
            loss_metrics = criterion(predicted_metrics, target_metrics)
            loss = config.SPECTRA_LOSS_WEIGHT * loss_spectra + config.METRIC_LOSS_WEIGHT * loss_metrics

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), spec_loss=loss_spectra.item(), met_loss=loss_metrics.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                struct_params = batch['struct'].to(config.DEVICE)
                target_spectra = batch['spectra'].to(config.DEVICE)
                target_metrics = batch['metrics'].to(config.DEVICE)

                predicted_spectra, predicted_metrics = model(struct_params)
                loss_spectra = criterion(predicted_spectra, target_spectra)
                loss_metrics = criterion(predicted_metrics, target_metrics)
                loss = config.SPECTRA_LOSS_WEIGHT * loss_spectra + config.METRIC_LOSS_WEIGHT * loss_metrics
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth'))
            print(f"Validation loss improved. Saving best model to {config.SAVED_MODELS_DIR}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    torch.save(model.state_dict(), os.path.join(config.SAVED_MODELS_DIR, 'final_forward_model.pth'))
    print("Finished pre-training simplified forward model.")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Simplified Forward Model Pre-training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(config.PLOT_DIR, 'forward_model_pretrain_loss_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Loss curve saved to {plot_path}")

if __name__ == '__main__':
    pretrain_forward_model()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse

from core.models.pinn import PINN

def pretrain_pinn(args):
    """
    Pre-trains the PINN model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    try:
        with np.load(args.data_path) as data:
            train_params = data['train_params']
            train_spectra = data['train_spectra']
            val_params = data['val_params']
            val_spectra = data['val_spectra']
    except FileNotFoundError:
        print(f"Error: Preprocessed data not found at {args.data_path}.")
        return

    train_dataset = TensorDataset(torch.from_numpy(train_params).float(), torch.from_numpy(train_spectra).float())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(val_params).float(), torch.from_numpy(val_spectra).float())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    pinn = PINN(params_dim=args.params_dim, spectra_dim=args.spectra_dim).to(device)
    optimizer = optim.Adam(pinn.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(device)

    print("PINN model, optimizer, and loss function initialized.")

    for epoch in range(args.n_epochs):
        pinn.train()
        train_loss = 0.0
        for params, spectra in train_loader:
            params, spectra = params.to(device), spectra.to(device)

            optimizer.zero_grad()
            predicted_spectra = pinn(params)
            loss = criterion(predicted_spectra, spectra)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        pinn.eval()
        val_loss = 0.0
        with torch.no_grad():
            for params, spectra in val_loader:
                params, spectra = params.to(device), spectra.to(device)
                predicted_spectra = pinn(params)
                loss = criterion(predicted_spectra, spectra)
                val_loss += loss.item()

        print(
            f"[Epoch {epoch}/{args.n_epochs}] "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}"
        )

        if epoch % args.save_interval == 0:
            os.makedirs("saved_models", exist_ok=True)
            torch.save(pinn.state_dict(), f"saved_models/pinn_pretrained_epoch_{epoch}.pth")
            print(f"Saved PINN model at epoch {epoch}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-train the PINN model.")
    
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use CUDA if available")
    parser.add_argument("--save_interval", type=int, default=10, help="interval between saving model checkpoints")
    parser.add_argument("--params_dim", type=int, default=4, help="dimensionality of the structural parameters")
    parser.add_argument("--spectra_dim", type=int, default=250, help="dimensionality of the spectra data")
    parser.add_argument("--data_path", type=str, default="dataset/preprocessed_data.npz", help="path to the preprocessed .npz data file")

    args = parser.parse_args()
    pretrain_pinn(args)

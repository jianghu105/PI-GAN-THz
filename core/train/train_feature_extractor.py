
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import argparse

from core.models.feature_extractor import FeatureExtractor

def train_feature_extractor(args):
    """
    Trains the FeatureExtractor model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    try:
        with np.load(args.data_path) as data:
            train_spectra = data['train_spectra']
            train_features = data['train_features']
            val_spectra = data['val_spectra']
            val_features = data['val_features']
    except FileNotFoundError:
        print(f"Error: Preprocessed data not found at {args.data_path}.")
        return

    train_dataset = TensorDataset(torch.from_numpy(train_spectra).float(), torch.from_numpy(train_features).float())
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = TensorDataset(torch.from_numpy(val_spectra).float(), torch.from_numpy(val_features).float())
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    feature_extractor = FeatureExtractor(spectra_dim=args.spectra_dim, features_dim=args.features_dim).to(device)
    optimizer = optim.Adam(feature_extractor.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(device)

    print("FeatureExtractor model, optimizer, and loss function initialized.")

    for epoch in range(args.n_epochs):
        feature_extractor.train()
        train_loss = 0.0
        for spectra, features in train_loader:
            spectra, features = spectra.to(device), features.to(device)

            optimizer.zero_grad()
            predicted_features = feature_extractor(spectra)
            loss = criterion(predicted_features, features)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        feature_extractor.eval()
        val_loss = 0.0
        with torch.no_grad():
            for spectra, features in val_loader:
                spectra, features = spectra.to(device), features.to(device)
                predicted_features = feature_extractor(spectra)
                loss = criterion(predicted_features, features)
                val_loss += loss.item()

        print(
            f"[Epoch {epoch}/{args.n_epochs}] "
            f"Train Loss: {train_loss / len(train_loader):.4f} | "
            f"Val Loss: {val_loss / len(val_loader):.4f}"
        )

        if epoch % args.save_interval == 0:
            os.makedirs("saved_models", exist_ok=True)
            torch.save(feature_extractor.state_dict(), f"saved_models/feature_extractor_epoch_{epoch}.pth")
            print(f"Saved FeatureExtractor model at epoch {epoch}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the FeatureExtractor model.")
    
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use CUDA if available")
    parser.add_argument("--save_interval", type=int, default=10, help="interval between saving model checkpoints")
    parser.add_argument("--spectra_dim", type=int, default=250, help="dimensionality of the spectra data")
    parser.add_argument("--features_dim", type=int, default=8, help="dimensionality of the resonance features")
    parser.add_argument("--data_path", type=str, default="dataset/preprocessed_data.npz", help="path to the preprocessed .npz data file")

    args = parser.parse_args()
    train_feature_extractor(args)

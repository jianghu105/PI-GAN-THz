
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import argparse

# Import models
from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.models.pinn import PINN
from core.models.feature_extractor import FeatureExtractor

# Import data preprocessing function if needed, though we assume data is preprocessed
# from core.utils.data_preprocessing import preprocess_data

def train(args):
    """
    Main training loop for the PI-GAN.
    """
    # --- 1. Setup and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Data ---
    # Load preprocessed data
    # For simplicity, we'll assume the preprocessed data is saved in a known location
    # In a real scenario, you might run preprocessing here if the data isn't found.
    try:
        with np.load(args.data_path) as data:
            train_params = data['train_params']
            train_spectra = data['train_spectra']
            train_features = data['train_features']
            val_params = data['val_params']
            val_spectra = data['val_spectra']
            val_features = data['val_features']
    except FileNotFoundError:
        print(f"Error: Preprocessed data not found at {args.data_path}.")
        print("Please run the data preprocessing script first.")
        return

    # Create PyTorch Datasets and DataLoaders
    train_dataset = TensorDataset(
        torch.from_numpy(train_params).float(),
        torch.from_numpy(train_spectra).float(),
        torch.from_numpy(train_features).float()
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # --- 3. Initialize Models ---
    generator = Generator(spectra_dim=args.spectra_dim, params_dim=args.params_dim, noise_dim=args.noise_dim).to(device)
    discriminator = Discriminator(spectra_dim=args.spectra_dim, params_dim=args.params_dim).to(device)
    pinn = PINN(params_dim=args.params_dim, spectra_dim=args.spectra_dim).to(device)
    feature_extractor = FeatureExtractor(spectra_dim=args.spectra_dim, features_dim=8).to(device)

    # Load pre-trained PINN model
    if args.pretrained_pinn_path:
        try:
            pinn.load_state_dict(torch.load(args.pretrained_pinn_path, map_location=device))
            print(f"Successfully loaded pre-trained PINN model from {args.pretrained_pinn_path}")
        except FileNotFoundError:
            print(f"Warning: Pre-trained PINN model not found at {args.pretrained_pinn_path}. Training from scratch.")

    # Load pre-trained FeatureExtractor model
    if args.pretrained_feature_extractor_path:
        try:
            feature_extractor.load_state_dict(torch.load(args.pretrained_feature_extractor_path, map_location=device))
            print(f"Successfully loaded pre-trained FeatureExtractor model from {args.pretrained_feature_extractor_path}")
        except FileNotFoundError:
            print(f"Warning: Pre-trained FeatureExtractor model not found at {args.pretrained_feature_extractor_path}. Training without feature loss.")

    # --- 4. Optimizers and Loss Functions ---
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_g, betas=(args.b1, args.b2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.b1, args.b2))
    
    # Loss functions
    adversarial_loss = nn.BCELoss().to(device)
    physics_loss = nn.MSELoss().to(device)
    feature_loss = nn.MSELoss().to(device)

    print("Models, optimizers, and loss functions initialized.")

    # --- 5. Training Loop ---
    for epoch in range(args.n_epochs):
        for i, (real_params, real_spectra, real_features) in enumerate(train_loader):

            # Move data to the selected device
            real_params = real_params.to(device)
            real_spectra = real_spectra.to(device)
            real_features = real_features.to(device)

            # Adversarial ground truths
            valid = torch.full((real_params.size(0), 1), 1.0, device=device, requires_grad=False)
            fake = torch.full((real_params.size(0), 1), 0.0, device=device, requires_grad=False)

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Sample noise
            z = torch.randn(real_params.size(0), args.noise_dim, device=device)

            # Generate a batch of parameters
            gen_params = generator(real_spectra, z)

            # Predict spectra from generated parameters using PINN
            gen_spectra_pinn = pinn(gen_params)

            # Predict features from the PINN-generated spectra
            gen_features = feature_extractor(gen_spectra_pinn)

            # Calculate losses
            g_loss_adv = adversarial_loss(discriminator(real_spectra, gen_params), valid)
            g_loss_phy = physics_loss(gen_spectra_pinn, real_spectra)
            g_loss_feat = feature_loss(gen_features, real_features)
            
            # Total Generator loss
            g_loss = g_loss_adv + args.lambda_phy * g_loss_phy + args.lambda_feat * g_loss_feat

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss for real data
            real_loss = adversarial_loss(discriminator(real_spectra, real_params), valid)

            # Loss for fake data
            fake_loss = adversarial_loss(discriminator(real_spectra, gen_params.detach()), fake)

            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        # --- 6. Logging and Saving ---
        print(
            f"[Epoch {epoch}/{args.n_epochs}] "
            f"[D loss: {d_loss.item():.4f}] "
            f"[G loss: {g_loss.item():.4f}] "
            f"(Adv: {g_loss_adv.item():.4f}, Phy: {g_loss_phy.item():.4f})"
        )

        if epoch % args.save_interval == 0:
            os.makedirs("saved_models", exist_ok=True)
            torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch}.pth")
            torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch_{epoch}.pth")
            print(f"Saved models at epoch {epoch}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the PI-GAN model for THz Metamaterial Inverse Design.")
    
    # Training parameters
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr_g", type=float, default=0.0002, help="adam: learning rate for generator")
    parser.add_argument("--lr_d", type=float, default=0.0002, help="adam: learning rate for discriminator")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use CUDA if available")
    parser.add_argument("--save_interval", type=int, default=50, help="interval between saving model checkpoints")

    # Model and Data parameters
    parser.add_argument("--spectra_dim", type=int, default=250, help="dimensionality of the spectra data")
    parser.add_argument("--params_dim", type=int, default=4, help="dimensionality of the structural parameters")
    parser.add_argument("--noise_dim", type=int, default=100, help="dimensionality of the noise vector")
    parser.add_argument("--data_path", type=str, default="dataset/preprocessed_data.npz", help="path to the preprocessed .npz data file")
    parser.add_argument("--pretrained_pinn_path", type=str, default="saved_models/pinn_pretrained_epoch_90.pth", help="path to the pre-trained PINN model")
    parser.add_argument("--pretrained_feature_extractor_path", type=str, default="saved_models/feature_extractor_epoch_90.pth", help="path to the pre-trained FeatureExtractor model")

    # Loss weights
    parser.add_argument("--lambda_phy", type=float, default=10.0, help="weight for the physics loss")
    parser.add_argument("--lambda_feat", type=float, default=1.0, help="weight for the feature matching loss")

    args = parser.parse_args()
    
    # Before training, we need to convert the split data from the preprocessing script into a single .npz file.
    # This is a placeholder for that logic.
    # For now, we will manually create this file from the output of the previous step.

    train(args)

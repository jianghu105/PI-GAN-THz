
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.utils.data_loader import get_dataloaders
from core.models.forward_model import ForwardModel
from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.utils.loss import PhysicsInformedLoss, gradient_penalty

def train_pigan():
    """Trains the PI-GAN model."""
    print("Starting PI-GAN training...")

    # Create directories if they don't exist
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Get DataLoaders and scalers
    train_loader, _, _, scalers = get_dataloaders(batch_size=config.BATCH_SIZE)

    # Initialize models
    # Load pre-trained forward model and freeze its parameters
    forward_model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS)
    ).to(config.DEVICE)
    forward_model.load_state_dict(torch.load(os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')))
    for param in forward_model.parameters():
        param.requires_grad = False
    forward_model.eval() # Set to eval mode

    generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS)
    ).to(config.DEVICE)

    discriminator = Discriminator(
        struct_dim=len(config.STRUCT_PARAMS),
        spectra_dim=len(config.SPECTRA_PARAMS),
        metric_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config.G_LR, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.D_LR, betas=(0.5, 0.9))

    # Loss functions
    # PhysicsInformedLoss takes the forward_model to calculate physics loss
    pi_loss_G = PhysicsInformedLoss(forward_model=forward_model,
                                    lambda_physics=config.LAMBDA_PHYSICS,
                                    lambda_metric=config.LAMBDA_METRIC)

    # Training loop
    for epoch in range(config.GAN_EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"GAN Epoch {epoch+1}/{config.GAN_EPOCHS}", leave=False)
        for i, batch in enumerate(progress_bar):
            real_struct = batch['struct'].to(config.DEVICE)
            real_spectra = batch['spectra'].to(config.DEVICE)
            real_metrics = batch['metrics'].to(config.DEVICE)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real samples
            D_real = discriminator(torch.cat([real_struct, real_spectra, real_metrics], dim=1))
            D_real_loss = -torch.mean(D_real)

            # Fake samples
            z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
            fake_struct = generator(z)
            
            # Get fake spectra from the frozen forward model
            with torch.no_grad():
                fake_spectra, fake_metrics = forward_model(fake_struct) # _ to ignore fake_metrics

            D_fake = discriminator(torch.cat([fake_struct.detach(), fake_spectra.detach(), fake_metrics.detach()], dim=1))
            D_fake_loss = torch.mean(D_fake)

            # Gradient penalty
            gp = gradient_penalty(discriminator, 
                                  torch.cat([real_struct, real_spectra, real_metrics], dim=1),
                                  torch.cat([fake_struct, fake_spectra, fake_metrics], dim=1))

            D_loss = D_real_loss + D_fake_loss + config.LAMBDA_GP * gp
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0) # Gradient clipping
            optimizer_D.step()

            # --- Train Generator (every N discriminator steps) ---
            if i % 5 == 0: # Train Generator less frequently
                optimizer_G.zero_grad()

                z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
                fake_struct = generator(z)
                
                # Get fake spectra from the frozen forward model
                # No need for no_grad() here as we need gradients for physics loss
                fake_spectra, fake_metrics = forward_model(fake_struct)

                D_fake_for_G = discriminator(torch.cat([fake_struct, fake_spectra, fake_metrics], dim=1))
                
                # Calculate generator loss using PhysicsInformedLoss
                G_loss, adv_loss, phys_loss, met_loss = pi_loss_G(
                    fake_struct, real_spectra, real_metrics, D_fake_for_G # Pass None for target_metrics
                )
                G_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0) # Gradient clipping
                optimizer_G.step()

                progress_bar.set_postfix(
                    D_loss=D_loss.item(), G_loss=G_loss.item(),
                    Adv_L=adv_loss.item(), Phys_L=phys_loss.item(), Met_L=met_loss.item()
                )
            else:
                progress_bar.set_postfix(D_loss=D_loss.item())

        print(f"Epoch {epoch+1}: D_Loss: {D_loss.item():.4f}, G_Loss: {G_loss.item():.4f}, "
              f"Adv_Loss: {adv_loss.item():.4f}, Phys_Loss: {phys_loss.item():.4f}, Met_Loss: {met_loss.item():.4f}")

        # Save models periodically
        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(config.SAVED_MODELS_DIR, f'generator_epoch_{epoch+1}.pth'))
            torch.save(discriminator.state_dict(), os.path.join(config.SAVED_MODELS_DIR, f'discriminator_epoch_{epoch+1}.pth'))
            print(f"Saved models at epoch {epoch+1}")

    print("Finished PI-GAN training.")
    # Save final models
    torch.save(generator.state_dict(), os.path.join(config.SAVED_MODELS_DIR, 'final_generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(config.SAVED_MODELS_DIR, 'final_discriminator.pth'))

if __name__ == '__main__':
    train_pigan()

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
from core.utils.loss import gradient_penalty, PhysicsInformedLoss

def train_pigan():
    """Trains the Physics-Informed Conditional GAN (cGAN)."""
    print("Starting Physics-Informed cGAN training...")

    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)

    train_loader, _, _, scalers = get_dataloaders(batch_size=config.BATCH_SIZE)
    metrics_scaler = scalers['metrics']

    # 1. Load pre-trained forward model (our physics engine)
    forward_model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS),
        metrics_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)
    forward_model.load_state_dict(torch.load(os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')))
    for param in forward_model.parameters():
        param.requires_grad = False
    forward_model.eval()

    # 2. Initialize Conditional Generator and Discriminator
    generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS),
        condition_dim=config.CONDITION_DIM
    ).to(config.DEVICE)

    discriminator = Discriminator(
        struct_dim=len(config.STRUCT_PARAMS),
        spectra_dim=len(config.SPECTRA_PARAMS),
        condition_dim=config.CONDITION_DIM
    ).to(config.DEVICE)

    optimizer_G = optim.Adam(generator.parameters(), lr=config.G_LR, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.D_LR, betas=(0.5, 0.9))
    
    physics_criterion = PhysicsInformedLoss(forward_model=forward_model, metrics_scaler=metrics_scaler).to(config.DEVICE)

    # --- Training Loop ---
    for epoch in range(config.GAN_EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"cGAN Epoch {epoch+1}/{config.GAN_EPOCHS}", leave=False)
        for i, batch in enumerate(progress_bar):
            real_struct = batch['struct'].to(config.DEVICE)
            real_spectra = batch['spectra'].to(config.DEVICE)
            condition = batch['metrics'].to(config.DEVICE) # Use metrics as condition

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real samples
            real_combined = torch.cat([real_struct, real_spectra, condition], dim=1)
            D_real_score, D_real_phys_err = discriminator(real_combined)
            D_real_loss = -torch.mean(D_real_score)

            # Fake samples
            z = torch.randn(real_struct.size(0), config.LATENT_DIM).to(config.DEVICE)
            fake_struct = generator(z, condition)
            with torch.no_grad():
                fake_spectra, _ = forward_model(fake_struct)

            fake_combined = torch.cat([fake_struct.detach(), fake_spectra.detach(), condition], dim=1)
            D_fake_score, D_fake_phys_err = discriminator(fake_combined)
            D_fake_loss = torch.mean(D_fake_score)

            # Physical error loss for discriminator
            target_real_phys_err = torch.zeros_like(D_real_phys_err)
            target_fake_phys_err = torch.mean((fake_spectra - real_spectra)**2, dim=1, keepdim=True)
            D_phys_err_loss = nn.MSELoss()(D_real_phys_err, target_real_phys_err) + \
                              nn.MSELoss()(D_fake_phys_err, target_fake_phys_err.detach())

            gp = gradient_penalty(discriminator, real_combined, fake_combined)
            D_loss = D_real_loss + D_fake_loss + config.LAMBDA_GP * gp + D_phys_err_loss

            D_loss.backward()
            optimizer_D.step()

            # --- Train Generator ---
            if i % 5 == 0:
                optimizer_G.zero_grad()

                z = torch.randn(real_struct.size(0), config.LATENT_DIM).to(config.DEVICE)
                fake_struct = generator(z, condition)
                fake_spectra, _ = forward_model(fake_struct)

                # Adversarial Loss
                fake_combined_for_G = torch.cat([fake_struct, fake_spectra, condition], dim=1)
                G_fake_score, G_physical_error_feedback = discriminator(fake_combined_for_G)

                # Physics-Informed Loss (Consistency Loss)
                G_loss, G_adv_loss, physics_loss, metric_loss, pid_feedback_loss, tv_loss = physics_criterion(
                    generated_struct=fake_struct,
                    target_spectra=real_spectra,
                    target_metrics=condition, # Assuming condition is target_metrics
                    D_fake_real_score=G_fake_score,
                    physical_error_feedback=G_physical_error_feedback
                )

                G_loss.backward()
                optimizer_G.step()

                progress_bar.set_postfix(D_loss=D_loss.item(), G_loss=G_loss.item(), Phys_Loss=physics_loss.item(), TV_Loss=tv_loss.item())

        if (epoch + 1) % 50 == 0:
            torch.save(generator.state_dict(), os.path.join(config.SAVED_MODELS_DIR, f'cgan_generator_epoch_{epoch+1}.pth'))
            print(f"Saved conditional generator model at epoch {epoch+1}")

    print("Finished Physics-Informed cGAN training.")
    torch.save(generator.state_dict(), os.path.join(config.SAVED_MODELS_DIR, 'final_cgan_generator.pth'))

if __name__ == '__main__':
    train_pigan()
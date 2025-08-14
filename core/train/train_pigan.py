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
from core.utils.loss import PhysicsInformedLoss, gradient_penalty, r1_regularization, mode_seeking_loss

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

    # Initialize conditional generator
    generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS),
        condition_dim=config.CONDITION_DIM # Pass condition_dim
    ).to(config.DEVICE)

    discriminator = Discriminator(
        struct_dim=len(config.STRUCT_PARAMS),
        spectra_dim=len(config.SPECTRA_PARAMS),
        metric_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=config.G_LR, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.D_LR, betas=(0.5, 0.9))

    # Loss functions (PID weight will be annealed each epoch)
    pi_loss_G = PhysicsInformedLoss(forward_model=forward_model,
                                    lambda_physics=config.LAMBDA_PHYSICS,
                                    lambda_metric=config.LAMBDA_METRIC,
                                    lambda_pid=config.LAMBDA_PID_FEEDBACK,
                                    metrics_scaler=scalers['metrics'])

    # Training loop
    for epoch in range(config.GAN_EPOCHS):
        # Anneal PID feedback weight linearly from initial to target over PID_FEEDBACK_ANNEAL_EPOCHS
        if config.PID_FEEDBACK_ANNEAL_EPOCHS > 0:
            t = min(1.0, (epoch + 1) / config.PID_FEEDBACK_ANNEAL_EPOCHS)
            current_pid = config.LAMBDA_PID_FEEDBACK + t * (config.PID_FEEDBACK_TARGET - config.LAMBDA_PID_FEEDBACK)
            pi_loss_G.lambda_pid = current_pid
        else:
            current_pid = pi_loss_G.lambda_pid
        progress_bar = tqdm(train_loader, desc=f"GAN Epoch {epoch+1}/{config.GAN_EPOCHS}", leave=False)
        for i, batch in enumerate(progress_bar):
            real_struct = batch['struct'].to(config.DEVICE)
            real_spectra = batch['spectra'].to(config.DEVICE)
            real_metrics = batch['metrics'].to(config.DEVICE)

            # --- Train Discriminator ---
            optimizer_D.zero_grad()

            # Real samples
            D_real_score, _ = discriminator(torch.cat([real_struct, real_spectra, real_metrics], dim=1))
            D_real_loss = -torch.mean(D_real_score)

            # Fake samples
            z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
            # Use real_metrics as condition for generator
            condition_for_G = real_metrics # Generator will try to produce struct for these metrics
            fake_struct = generator(z, condition_for_G) # Pass condition
            
            # Get fake spectra from the frozen forward model
            with torch.no_grad():
                fake_spectra, fake_metrics = forward_model(fake_struct) 

            D_fake_score, _ = discriminator(torch.cat([fake_struct.detach(), fake_spectra.detach(), fake_metrics.detach()], dim=1))
            D_fake_loss = torch.mean(D_fake_score)

            # Gradient penalty
            gp = gradient_penalty(discriminator, 
                                  torch.cat([real_struct, real_spectra, real_metrics], dim=1),
                                  torch.cat([fake_struct, fake_spectra, fake_metrics], dim=1))

            D_loss = D_real_loss + D_fake_loss + config.LAMBDA_GP * gp

            # Optional R1 regularization on real samples (set LAMBDA_R1>0 to enable)
            if config.LAMBDA_R1 > 0.0:
                r1 = r1_regularization(discriminator, torch.cat([real_struct, real_spectra, real_metrics], dim=1))
                D_loss = D_loss + (config.LAMBDA_R1 * 0.5) * r1
            D_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0) # Gradient clipping
            optimizer_D.step()

            # --- Train Generator (every N discriminator steps) ---
            if i % 5 == 0: # Train Generator less frequently
                optimizer_G.zero_grad()

                z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
                # Use real_metrics as condition for generator
                condition_for_G = real_metrics
                fake_struct = generator(z, condition_for_G) # Pass condition
                
                # Get fake spectra from the frozen forward model
                fake_spectra, fake_metrics = forward_model(fake_struct)

                D_fake_for_G_score, physical_error_feedback = discriminator(torch.cat([fake_struct, fake_spectra, fake_metrics], dim=1))
                
                # Calculate generator loss using PhysicsInformedLoss
                # Pass physical_error_feedback to pi_loss_G
                G_loss, adv_loss, phys_loss, met_loss, pid_feedback_loss = pi_loss_G(
                    fake_struct, real_spectra, real_metrics, D_fake_for_G_score, physical_error_feedback,
                    predicted_spectra=fake_spectra, predicted_metrics=fake_metrics
                )
                # Mode-seeking diversity loss: encourage diverse structures for different latents under same condition
                if config.LAMBDA_MODE_SEEKING > 0.0:
                    z2 = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
                    fake_struct_2 = generator(z2, condition_for_G)
                    ms_loss = mode_seeking_loss(fake_struct, fake_struct_2, z, z2)
                    G_loss = G_loss + config.LAMBDA_MODE_SEEKING * ms_loss
                else:
                    ms_loss = torch.tensor(0.0, device=config.DEVICE)
                G_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0) # Gradient clipping
                optimizer_G.step()

                progress_bar.set_postfix(
                    D_loss=D_loss.item(), G_loss=G_loss.item(),
                    Adv_L=adv_loss.item(), Phys_L=phys_loss.item(), Met_L=met_loss.item(),
                    PID_L=pid_feedback_loss.item(), PID_w=current_pid, MS_L=ms_loss.item()
                )
            else:
                progress_bar.set_postfix(D_loss=D_loss.item(), PID_w=current_pid)

        # Guard against epochs where G was not updated in the last iteration
        try:
            print(f"Epoch {epoch+1}: D_Loss: {D_loss.item():.4f}, G_Loss: {G_loss.item():.4f}, "
                  f"Adv_Loss: {adv_loss.item():.4f}, Phys_Loss: {phys_loss.item():.4f}, Met_Loss: {met_loss.item():.4f}, "
                  f"PID_Loss: {pid_feedback_loss.item():.4f}, PID_w: {current_pid:.4f}")
        except UnboundLocalError:
            print(f"Epoch {epoch+1}: D_Loss: {D_loss.item():.4f}, PID_w: {current_pid:.4f} (G not updated this epoch's last step)")

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
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config

class PhysicsInformedLoss(nn.Module):
    """Computes the physics-informed loss for the GAN.

    This loss consists of four parts:
    1.  Adversarial Loss: How well the generator fools the discriminator.
    2.  Physics Loss: MSE between predicted spectra and target spectra (both normalized to [0,1]).
    3.  Metric Loss: MSE between normalized predicted metrics and target normalized metrics.
        Predicted metrics are affinely normalized using the train-fitted MinMax scaler parameters.
    4.  Physical Error Feedback Loss: Feedback from the discriminator's physical error head.
    """
    def __init__(self, forward_model, lambda_physics=config.LAMBDA_PHYSICS, 
                 lambda_metric=config.LAMBDA_METRIC, lambda_pid=config.LAMBDA_PID_FEEDBACK,
                 metrics_scaler=None):
        super().__init__()
        self.forward_model = forward_model
        self.lambda_physics = lambda_physics
        self.lambda_metric = lambda_metric
        self.lambda_pid = lambda_pid
        self.mse_loss = nn.MSELoss()
        # Extract affine params from sklearn MinMaxScaler: x_scaled = x * scale_ + min_
        if metrics_scaler is not None:
            # Register as buffers so they move with .to(device)
            scale = torch.tensor(metrics_scaler.scale_, dtype=torch.float32)
            offset = torch.tensor(metrics_scaler.min_, dtype=torch.float32)
            self.register_buffer('metrics_scale', scale)
            self.register_buffer('metrics_offset', offset)
        else:
            self.register_buffer('metrics_scale', torch.ones(len(config.METRIC_PARAMS), dtype=torch.float32))
            self.register_buffer('metrics_offset', torch.zeros(len(config.METRIC_PARAMS), dtype=torch.float32))

    def forward(self, generated_struct, target_spectra, target_metrics, D_fake_real_score, physical_error_feedback,
                predicted_spectra: torch.Tensor = None, predicted_metrics: torch.Tensor = None):
        # 1. Adversarial Loss (from generator's perspective - wants to maximize D's output)
        adversarial_loss = -torch.mean(D_fake_real_score)

        # 2. Physics-informed Loss
        if predicted_spectra is None or predicted_metrics is None:
            predicted_spectra, predicted_metrics = self.forward_model(generated_struct)

        physics_loss = self.mse_loss(predicted_spectra, target_spectra)

        # 3. Metric Loss with scaler normalization
        if self.lambda_metric > 0 and target_metrics is not None:
            predicted_metrics_cleaned = torch.nan_to_num(predicted_metrics, nan=0.0)
            # Apply affine normalization using train-fitted scaler: scaled = x * scale + offset
            scale = self.metrics_scale.to(predicted_metrics_cleaned.device)
            offset = self.metrics_offset.to(predicted_metrics_cleaned.device)
            predicted_metrics_scaled = predicted_metrics_cleaned * scale + offset
            target_metrics_cleaned = torch.nan_to_num(target_metrics, nan=0.0)
            metric_loss = self.mse_loss(predicted_metrics_scaled, target_metrics_cleaned)
        else:
            metric_loss = torch.tensor(0.0, device=config.DEVICE)

        # 4. Physical Error Feedback Loss
        pid_feedback_loss = self.lambda_pid * torch.mean(physical_error_feedback)

        # Total Generator Loss
        total_loss = adversarial_loss + \
                     self.lambda_physics * physics_loss + \
                     self.lambda_metric * metric_loss + \
                     pid_feedback_loss

        return total_loss, adversarial_loss, physics_loss, metric_loss, pid_feedback_loss

def mode_seeking_loss(g_output_a: torch.Tensor, g_output_b: torch.Tensor, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
    """Mode-seeking regularization (MS-GAN): maximize ratio of output difference to latent difference.
    Implemented as negative ratio to be minimized: - ||G(z1)-G(z2)||_1 / (||z1 - z2||_1 + eps).
    """
    eps = config.MODE_SEEKING_EPS
    numerator = torch.mean(torch.abs(g_output_a - g_output_b))
    denominator = torch.mean(torch.abs(z_a - z_b)) + eps
    return -(numerator / denominator)

def gradient_penalty(discriminator, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN-GP."""
    batch_size = real_samples.size(0)
    # Broadcast alpha to the full feature dimension to avoid ambiguity
    alpha = torch.rand(batch_size, 1).to(config.DEVICE)
    alpha = alpha.expand_as(real_samples)

    interpolated = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    # Discriminator now returns two outputs, we only need the real_fake_score for GP
    d_interpolated, _ = discriminator(interpolated)

    grad_outputs = torch.ones_like(d_interpolated).to(config.DEVICE)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = torch.mean((gradient_norm - 1) ** 2)
    return gp

def r1_regularization(discriminator, real_samples):
    """R1 regularization term on real samples: (gamma/2) * E[||grad D(x)||^2].
    Returns per-batch mean of squared gradients of the real/fake head.
    """
    real_samples.requires_grad_(True)
    real_scores, _ = discriminator(real_samples)
    grad_outputs = torch.ones_like(real_scores)
    gradients = torch.autograd.grad(
        outputs=real_scores,
        inputs=real_samples,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    r1 = torch.mean(torch.sum(gradients ** 2, dim=1))
    return r1
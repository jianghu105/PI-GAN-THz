import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config

# 新增的物理约束模块
class PhysicalConstraintModule(nn.Module):
    """Calculates a penalty if structural parameters are outside defined bounds."""
    def __init__(self):
        super().__init__()
        # IMPORTANT: Ensure config.STRUCT_MIN_BOUNDS and config.STRUCT_MAX_BOUNDS are defined in config.py
        # Example placeholder if not defined:
        # self.min_bounds = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        # self.max_bounds = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
        
        # Assuming config.STRUCT_MIN_BOUNDS and config.STRUCT_MAX_BOUNDS are lists/tuples
        self.min_bounds = torch.tensor(config.STRUCT_MIN_BOUNDS, dtype=torch.float32)
        self.max_bounds = torch.tensor(config.STRUCT_MAX_BOUNDS, dtype=torch.float32)

    def forward(self, struct_params):
        # Ensure bounds tensor is on the same device as input
        min_bounds = self.min_bounds.to(struct_params.device)
        max_bounds = self.max_bounds.to(struct_params.device)

        # Calculate violation for values below min_bounds and above max_bounds
        lower_violation = torch.relu(min_bounds - struct_params) # max(0, min_bounds - value)
        upper_violation = torch.relu(struct_params - max_bounds) # max(0, value - max_bounds)
        
        # Sum violations across all structural parameters for each sample
        penalty = torch.sum(lower_violation + upper_violation, dim=1)
        return penalty # Returns a tensor of shape (batch_size,)

class Discriminator(nn.Module):
    """Multi-branch discriminator that processes structure, spectra, and metrics with dedicated encoders,
    then fuses them for real/fake scoring and physical error feedback."""
    def __init__(self, struct_dim, spectra_dim, metric_dim):
        super().__init__()

        self.struct_dim = struct_dim

        # Structure branch (MLP)
        self.struct_encoder = nn.Sequential(
            spectral_norm(nn.Linear(struct_dim, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 128)),
            nn.LeakyReLU(0.2),
        )

        # Spectra branch (1D CNN)
        self.spectra_encoder = nn.Sequential(
            spectral_norm(nn.Conv1d(1, 64, kernel_size=5, padding=2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv1d(64, 128, kernel_size=5, padding=2)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv1d(128, 256, kernel_size=3, padding=1)),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
        )

        # Metric branch (MLP)
        self.metric_encoder = nn.Sequential(
            spectral_norm(nn.Linear(metric_dim, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 128)),
            nn.LeakyReLU(0.2),
        )

        # Calculate fusion dim (spectra branch outputs 256*16)
        fusion_dim = 128 + (256 * 16) + 128

        # Fusion + heads
        self.fusion = nn.Sequential(
            spectral_norm(nn.Linear(fusion_dim, 256)),
            nn.LeakyReLU(0.2),
        )
        self.real_fake_head = nn.Sequential(
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 1))
        )
        self.learned_physical_error_head = nn.Sequential(
            spectral_norm(nn.Linear(256, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, 1))
        )

        # Physical Constraint Module (Non-trainable)
        self.physical_constraint_module = PhysicalConstraintModule()

    def forward(self, combined_input):
        # Split combined input
        struct = combined_input[:, :self.struct_dim]
        spectra = combined_input[:, self.struct_dim:-8]  # assume last 8 are metrics
        metrics = combined_input[:, -8:]

        # Encoders
        struct_feat = self.struct_encoder(struct)
        spectra_feat = self.spectra_encoder(spectra.unsqueeze(1))
        metric_feat = self.metric_encoder(metrics)

        fused = torch.cat([struct_feat, spectra_feat, metric_feat], dim=1)
        fused = self.fusion(fused)

        real_fake_score = self.real_fake_head(fused)
        learned_physical_error = self.learned_physical_error_head(fused).squeeze(1)

        # Rule-based penalty
        rule_based_penalty = self.physical_constraint_module(struct)
        physical_error_feedback = learned_physical_error + rule_based_penalty
        return real_fake_score, physical_error_feedback

if __name__ == '__main__':
    # Example usage for Discriminator
    # For testing, let's define dummy values if config is not fully available
    class DummyConfig:
        STRUCT_PARAMS = ['r1', 'r2', 'w', 'g']
        SPECTRA_PARAMS = [f'Freq_{i}' for i in range(250)]
        METRIC_PARAMS = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
        BATCH_SIZE = 16
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Placeholder for structural parameter bounds
        STRUCT_MIN_BOUNDS = [0.0, 0.0, 0.0, 0.0] # Example: all params min 0
        STRUCT_MAX_BOUNDS = [1.0, 1.0, 1.0, 1.0] # Example: all params max 1
    
    # Temporarily override config for testing if not running in full project context
    try:
        from config import config as actual_config
    except ImportError:
        print("Using dummy config for testing purposes.")
        config = DummyConfig()
    else:
        config = actual_config

    discriminator = Discriminator(
        struct_dim=len(config.STRUCT_PARAMS),
        spectra_dim=len(config.SPECTRA_PARAMS),
        metric_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)
    
    # Create dummy inputs (combined_input)
    # total_input_dim = struct_dim + spectra_dim + metric_dim
    dummy_combined_input = torch.randn(
        config.BATCH_SIZE,
        len(config.STRUCT_PARAMS) + len(config.SPECTRA_PARAMS) + len(config.METRIC_PARAMS)
    ).to(config.DEVICE)
    
    real_fake_score, physical_error_feedback = discriminator(dummy_combined_input)
    
    print("Discriminator Test (1D CNN + Physical Error Feedback)")
    print(f"Input combined shape: {dummy_combined_input.shape}")
    print(f"Real/Fake Score shape: {real_fake_score.shape}")
    print(f"Physical Error Feedback shape: {physical_error_feedback.shape}")
    print(f"Sample Real/Fake Score (first): {real_fake_score[0].item()}")
    print(f"Sample Physical Error Feedback (first): {physical_error_feedback[0].item()}")
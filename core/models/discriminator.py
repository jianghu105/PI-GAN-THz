import torch
import torch.nn as nn
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
    """Discriminates between real and fake (structure, spectra, metric) tuples,
    and provides physical error feedback."""
    def __init__(self, struct_dim, spectra_dim, metric_dim):
        super().__init__()
        
        self.struct_dim = struct_dim # Used to extract structural parameters from combined input
        total_input_dim = struct_dim + spectra_dim + metric_dim
        
        # 1D CNN Backbone
        # Input shape for Conv1d: (batch_size, in_channels, sequence_length)
        # Here, in_channels=1, sequence_length=total_input_dim
        self.cnn_backbone = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Flatten() # Flatten the output of CNN before passing to Linear layers
        )
        
        # Calculate the output dimension of the CNN backbone after flattening
        # Assuming padding=1, kernel_size=3, stride=1, the sequence length remains total_input_dim
        cnn_output_dim = 256 * total_input_dim 

        # Head 1: Real/Fake Score (Traditional GAN output)
        self.real_fake_head = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1) # No sigmoid, for WGAN-GP
        )

        # Head 2: Learned Physical Error Feedback
        self.learned_physical_error_head = nn.Sequential(
            nn.Linear(cnn_output_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1) # Outputs a scalar physical error score
        )
        
        # Physical Constraint Module (Non-trainable)
        self.physical_constraint_module = PhysicalConstraintModule()

    def forward(self, combined_input):
        # combined_input shape: (batch_size, total_input_dim)
        # Reshape for 1D CNN: (batch_size, 1, total_input_dim)
        x = combined_input.unsqueeze(1)
        
        # Extract features through the 1D CNN backbone
        cnn_features = self.cnn_backbone(x)
        
        # Calculate the real/fake score
        real_fake_score = self.real_fake_head(cnn_features)

        # Calculate the learned physical error
        # Squeeze to get shape (batch_size,) from (batch_size, 1)
        learned_physical_error = self.learned_physical_error_head(cnn_features).squeeze(1)

        # Extract structural parameters from the combined_input for rule-based physical constraint check
        # Assuming struct_dim is the first part of the combined input
        struct_params = combined_input[:, :self.struct_dim]
        rule_based_penalty = self.physical_constraint_module(struct_params)
        
        # Combine learned physical error and rule-based penalty
        # This is a design choice; simple addition is used here.
        physical_error_feedback = learned_physical_error + rule_based_penalty
        
        # Return both the real/fake score and the physical error feedback
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
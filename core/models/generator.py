import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config

class Generator(nn.Module):
    """Generates structural parameters from a latent vector and a condition vector."""
    def __init__(self, latent_dim, output_dim, condition_dim):
        super().__init__()
        
        # Input dimension is latent_dim + condition_dim
        input_dim = latent_dim + condition_dim

        # Deeper and wider network for cGAN
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # To ensure outputs are in [0, 1] range, matching the scaled data
        )

    def forward(self, z, condition):
        # Concatenate latent vector and condition vector
        combined_input = torch.cat([z, condition], dim=1)
        output = self.network(combined_input)
        assert not torch.isnan(output).any(), "Generator output contains NaN values!"
        return output

if __name__ == '__main__':
    # Example usage
    class DummyConfig:
        LATENT_DIM = 100
        STRUCT_PARAMS = ['r1', 'r2', 'w', 'g']
        # Define CONDITION_DIM based on what you want to condition on (e.g., len(SPECTRA_PARAMS) or len(METRIC_PARAMS))
        # For testing, let's assume conditioning on metrics (8 dimensions)
        METRIC_PARAMS = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
        CONDITION_DIM = len(METRIC_PARAMS) # Example: conditioning on metrics
        BATCH_SIZE = 16
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Temporarily override config for testing if not running in full project context
    try:
        from config import config as actual_config
    except ImportError:
        print("Using dummy config for testing purposes.")
        config = DummyConfig()
    else:
        config = actual_config

    generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS),
        condition_dim=config.CONDITION_DIM
    ).to(config.DEVICE)
    
    # Create a dummy latent vector and a dummy condition vector
    dummy_z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
    dummy_condition = torch.randn(config.BATCH_SIZE, config.CONDITION_DIM).to(config.DEVICE)
    
    generated_params = generator(dummy_z, dummy_condition)
    
    print("Conditional Generator Test")
    print(f"Input latent vector shape: {dummy_z.shape}")
    print(f"Input condition vector shape: {dummy_condition.shape}")
    print(f"Generated parameters shape: {generated_params.shape}")

import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config

class Generator(nn.Module):
    """Generates structural parameters from a latent vector."""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Sigmoid()  # To ensure outputs are in [0, 1] range, matching the scaled data
        )

    def forward(self, z):
        output = self.network(z)
        assert not torch.isnan(output).any(), "Generator output contains NaN values!"
        return output

if __name__ == '__main__':
    # Example usage
    generator = Generator(latent_dim=config.LATENT_DIM, output_dim=len(config.STRUCT_PARAMS)).to(config.DEVICE)
    
    # Create a dummy latent vector
    dummy_z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
    
    generated_params = generator(dummy_z)
    
    print("Generator Test")
    print(f"Input latent vector shape: {dummy_z.shape}")
    print(f"Generated parameters shape: {generated_params.shape}")


import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config

class Discriminator(nn.Module):
    """Discriminates between real and fake (structure, spectra, metric) tuples."""
    def __init__(self, struct_dim, spectra_dim, metric_dim):
        super().__init__()
        
        # The input to the discriminator is the concatenation of all three parts
        total_input_dim = struct_dim + spectra_dim + metric_dim
        
        self.network = nn.Sequential(
            nn.Linear(total_input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
            # No sigmoid at the end, as we are using WGAN-GP loss which expects raw scores
        )

    def forward(self, combined_input):
        return self.network(combined_input)

if __name__ == '__main__':
    # Example usage
    discriminator = Discriminator(
        struct_dim=len(config.STRUCT_PARAMS),
        spectra_dim=len(config.SPECTRA_PARAMS),
        metric_dim=len(config.METRIC_PARAMS)
    ).to(config.DEVICE)
    
    # Create dummy inputs
    dummy_struct = torch.randn(config.BATCH_SIZE, len(config.STRUCT_PARAMS)).to(config.DEVICE)
    dummy_spectra = torch.randn(config.BATCH_SIZE, len(config.SPECTRA_PARAMS)).to(config.DEVICE)
    dummy_metrics = torch.randn(config.BATCH_SIZE, len(config.METRIC_PARAMS)).to(config.DEVICE)
    
    output = discriminator(dummy_struct, dummy_spectra, dummy_metrics)
    
    print("Discriminator Test")
    print(f"Input struct shape:  {dummy_struct.shape}")
    print(f"Input spectra shape: {dummy_spectra.shape}")
    print(f"Input metrics shape: {dummy_metrics.shape}")
    print(f"Output shape:        {output.shape}")

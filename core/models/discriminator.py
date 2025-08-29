import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config

class Discriminator(nn.Module):
    """A conditional discriminator with a second head for physical error prediction."""
    def __init__(self, struct_dim, spectra_dim, condition_dim):
        super().__init__()

        total_input_dim = struct_dim + spectra_dim + condition_dim

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            spectral_norm(nn.Linear(total_input_dim, 512)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Head for real/fake classification
        self.real_fake_head = spectral_norm(nn.Linear(256, 1))

        # Head for physical error prediction
        self.physical_head = nn.Sequential(
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(128, 1))
        )

    def forward(self, combined_input):
        """The forward pass returns a real/fake score and a physical error prediction."""
        features = self.feature_extractor(combined_input)
        score = self.real_fake_head(features)
        physical_error = self.physical_head(features)
        return score, physical_error

if __name__ == '__main__':
    class DummyConfig:
        STRUCT_PARAMS = ['r1', 'r2', 'w', 'g']
        SPECTRA_PARAMS = [f'Freq_{i}' for i in range(250)]
        METRIC_PARAMS = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
        CONDITION_DIM = len(METRIC_PARAMS)
        BATCH_SIZE = 16
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = DummyConfig()

    discriminator = Discriminator(
        struct_dim=len(config.STRUCT_PARAMS),
        spectra_dim=len(config.SPECTRA_PARAMS),
        condition_dim=config.CONDITION_DIM
    ).to(config.DEVICE)
    
    dummy_combined_input = torch.randn(
        config.BATCH_SIZE,
        len(config.STRUCT_PARAMS) + len(config.SPECTRA_PARAMS) + config.CONDITION_DIM
    ).to(config.DEVICE)
    
    score, _ = discriminator(dummy_combined_input)
    
    print("Conditional Discriminator Test")
    print(f"Input combined shape: {dummy_combined_input.shape}")
    print(f"Score shape: {score.shape}")
    print(f"Sample Score (first): {score[0].item()}")
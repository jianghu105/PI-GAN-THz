

import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    A neural network to extract physical resonance features from a spectrum.

    Takes a transmission spectrum as input and predicts the 8 key
    resonance features (f1, f2, Q1, FoM1, S1, Q2, FoM2, S2).
    """
    def __init__(self, spectra_dim=250, features_dim=8, hidden_dim=256):
        """
        Initializes the FeatureExtractor model.

        Args:
            spectra_dim (int): The dimensionality of the input spectrum.
            features_dim (int): The dimensionality of the output resonance features.
            hidden_dim (int): The size of the hidden layers.
        """
        super(FeatureExtractor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(spectra_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, features_dim)
            # No output activation, as features are normalized and handled by loss.
        )

    def forward(self, spectrum):
        """
        Forward pass of the FeatureExtractor model.

        Args:
            spectrum (torch.Tensor): The transmission spectrum tensor. Shape: (batch_size, spectra_dim).

        Returns:
            torch.Tensor: The predicted resonance features. Shape: (batch_size, features_dim).
        """
        predicted_features = self.model(spectrum)
        return predicted_features

if __name__ == '__main__':
    # Example usage and model summary
    batch_size = 32
    spectra_dim = 250
    features_dim = 8

    # Create a FeatureExtractor instance
    feature_extractor = FeatureExtractor(spectra_dim=spectra_dim, features_dim=features_dim)
    print("--- FeatureExtractor Architecture ---")
    print(feature_extractor)

    # Create a dummy input tensor
    dummy_spectrum = torch.randn(batch_size, spectra_dim)

    # Get the output
    output_features = feature_extractor(dummy_spectrum)

    print("\n--- Input/Output Shapes ---")
    print(f"Input spectrum shape:  {dummy_spectrum.shape}")
    print(f"Output features shape: {output_features.shape}")
    assert output_features.shape == (batch_size, features_dim)
    print("\nSuccessfully tested the FeatureExtractor model.")


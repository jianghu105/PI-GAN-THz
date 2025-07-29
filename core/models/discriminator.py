

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    The Discriminator model for the PI-GAN.

    Takes a pair of (structural parameters, transmission spectrum) as input
    and outputs a single value indicating whether the pair is real or fake.
    """
    def __init__(self, spectra_dim=250, params_dim=4, hidden_dim=256):
        """
        Initializes the Discriminator model.

        Args:
            spectra_dim (int): The dimensionality of the input spectrum.
            params_dim (int): The dimensionality of the input structural parameters.
            hidden_dim (int): The size of the hidden layers.
        """
        super(Discriminator, self).__init__()
        self.input_dim = spectra_dim + params_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # To output a probability (real vs. fake)
        )

    def forward(self, spectrum, params):
        """
        Forward pass of the Discriminator.

        Args:
            spectrum (torch.Tensor): The transmission spectrum tensor. Shape: (batch_size, spectra_dim).
            params (torch.Tensor): The structural parameters tensor. Shape: (batch_size, params_dim).

        Returns:
            torch.Tensor: A single logit value for each input pair. Shape: (batch_size, 1).
        """
        # Concatenate spectrum and parameters along the feature dimension
        combined_input = torch.cat([spectrum, params], dim=1)
        validity = self.model(combined_input)
        return validity

if __name__ == '__main__':
    # Example usage and model summary
    batch_size = 32
    spectra_dim = 250
    params_dim = 4

    # Create a discriminator instance
    discriminator = Discriminator(spectra_dim=spectra_dim, params_dim=params_dim)
    print("--- Discriminator Architecture ---")
    print(discriminator)

    # Create dummy input tensors
    dummy_spectrum = torch.randn(batch_size, spectra_dim)
    dummy_params = torch.randn(batch_size, params_dim)

    # Get the output
    output_validity = discriminator(dummy_spectrum, dummy_params)

    print("\n--- Input/Output Shapes ---")
    print(f"Input spectrum shape: {dummy_spectrum.shape}")
    print(f"Input params shape:   {dummy_params.shape}")
    print(f"Output validity shape: {output_validity.shape}")
    assert output_validity.shape == (batch_size, 1)
    print("\nSuccessfully tested the Discriminator model.")


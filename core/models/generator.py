

import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    The Generator model for the PI-GAN.

    Takes a target transmission spectrum and a random noise vector as input
    and generates the corresponding structural parameters (r1, r2, w, g).
    """
    def __init__(self, spectra_dim=250, params_dim=4, noise_dim=100, hidden_dim=256):
        """
        Initializes the Generator model.

        Args:
            spectra_dim (int): The dimensionality of the input spectrum (e.g., 250 frequency points).
            params_dim (int): The dimensionality of the output structural parameters (e.g., 4 for r1, r2, w, g).
            noise_dim (int): The dimensionality of the random noise vector.
            hidden_dim (int): The size of the hidden layers.
        """
        super(Generator, self).__init__()
        self.input_dim = spectra_dim + noise_dim
        self.params_dim = params_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.params_dim),
            nn.Sigmoid()  # To ensure output is in [0, 1] as parameters are normalized
        )

    def forward(self, spectrum, noise):
        """
        Forward pass of the Generator.

        Args:
            spectrum (torch.Tensor): The target transmission spectrum tensor. Shape: (batch_size, spectra_dim).
            noise (torch.Tensor): The random noise tensor. Shape: (batch_size, noise_dim).

        Returns:
            torch.Tensor: The generated structural parameters. Shape: (batch_size, params_dim).
        """
        # Concatenate spectrum and noise along the feature dimension
        combined_input = torch.cat([spectrum, noise], dim=1)
        generated_params = self.model(combined_input)
        return generated_params

if __name__ == '__main__':
    # Example usage and model summary
    batch_size = 32
    spectra_dim = 250
    params_dim = 4
    noise_dim = 100

    # Create a generator instance
    generator = Generator(spectra_dim=spectra_dim, params_dim=params_dim, noise_dim=noise_dim)
    print("--- Generator Architecture ---")
    print(generator)

    # Create dummy input tensors
    dummy_spectrum = torch.randn(batch_size, spectra_dim)
    dummy_noise = torch.randn(batch_size, noise_dim)

    # Get the output
    output_params = generator(dummy_spectrum, dummy_noise)

    print("\n--- Input/Output Shapes ---")
    print(f"Input spectrum shape: {dummy_spectrum.shape}")
    print(f"Input noise shape:    {dummy_noise.shape}")
    print(f"Output params shape:  {output_params.shape}")
    assert output_params.shape == (batch_size, params_dim)
    print("\nSuccessfully tested the Generator model.")


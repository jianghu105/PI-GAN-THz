

import torch
import torch.nn as nn

class PINN(nn.Module):
    """
    The Physics-Informed Neural Network (PINN) forward model.

    Takes structural parameters (r1, r2, w, g) as input and predicts
    the corresponding transmission spectrum.
    """
    def __init__(self, params_dim=4, spectra_dim=250, hidden_dim=512):
        """
        Initializes the PINN model.

        Args:
            params_dim (int): The dimensionality of the input structural parameters.
            spectra_dim (int): The dimensionality of the output spectrum.
            hidden_dim (int): The size of the hidden layers.
        """
        super(PINN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(params_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, spectra_dim)
            # No activation on the output layer to allow for any value range,
            # as the data is normalized and can be handled by the loss function.
        )

    def forward(self, params):
        """
        Forward pass of the PINN model.

        Args:
            params (torch.Tensor): The structural parameters tensor. Shape: (batch_size, params_dim).

        Returns:
            torch.Tensor: The predicted transmission spectrum. Shape: (batch_size, spectra_dim).
        """
        predicted_spectrum = self.model(params)
        return predicted_spectrum

if __name__ == '__main__':
    # Example usage and model summary
    batch_size = 32
    params_dim = 4
    spectra_dim = 250


    # Create a PINN instance
    pinn = PINN(params_dim=params_dim, spectra_dim=spectra_dim)
    print("--- PINN Architecture ---")
    print(pinn)

    # Create a dummy input tensor
    dummy_params = torch.randn(batch_size, params_dim)

    # Get the output
    output_spectrum = pinn(dummy_params)

    print("\n--- Input/Output Shapes ---")
    print(f"Input params shape:  {dummy_params.shape}")
    print(f"Output spectrum shape: {output_spectrum.shape}")
    assert output_spectrum.shape == (batch_size, spectra_dim)
    print("\nSuccessfully tested the PINN model.")


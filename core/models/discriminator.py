# PI_GAN_THZ/core/models/discriminator.py

import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self, input_spec_dim, input_param_dim): # <--- Ensure these arguments are present
        """
        Discriminator for PI-GAN.
        Takes concatenated spectrum and structural parameters as input.

        Args:
            input_spec_dim (int): Dimension of the input spectrum (cfg.DISCRIMINATOR_INPUT_SPEC_DIM).
            input_param_dim (int): Dimension of the input structural parameters (cfg.DISCRIMINATOR_INPUT_PARAM_DIM).
        """
        super(Discriminator, self).__init__()
        
        # Calculate the total input dimension after concatenation
        total_input_dim = input_spec_dim + input_param_dim 
        
        self.main = nn.Sequential(
            nn.Linear(total_input_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1), # Output a single value for real/fake classification
            nn.Sigmoid() # Output probability that input is real
        )

    def forward(self, spectrum, params):
        # Concatenate spectrum and parameters along the feature dimension
        # Ensure spectrum and params are flat if they are not already (e.g., from CNN outputs)
        if spectrum.dim() > 2:
            spectrum = spectrum.view(spectrum.size(0), -1)
        if params.dim() > 2:
            params = params.view(params.size(0), -1)

        combined_input = torch.cat((spectrum, params), dim=1)
        return self.main(combined_input)
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config

class StructuralParamProjector(nn.Module):
    """Projects normalized structural parameters to respect bounds and relations.

    Operates in the normalized space expected by the rest of the pipeline.
    - Clamps/affine maps to [STRUCT_MIN_BOUNDS, STRUCT_MAX_BOUNDS]
    - Optionally enforces r1 >= r2 by sorting the first two parameters
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('min_bounds', torch.tensor(config.STRUCT_MIN_BOUNDS, dtype=torch.float32))
        self.register_buffer('max_bounds', torch.tensor(config.STRUCT_MAX_BOUNDS, dtype=torch.float32))
        self.enforce_r1_ge_r2 = getattr(config, 'ENFORCE_R1_GE_R2', True)

    def forward(self, normalized_params: torch.Tensor) -> torch.Tensor:
        min_b = self.min_bounds.to(normalized_params.device)
        max_b = self.max_bounds.to(normalized_params.device)

        projected = min_b + (max_b - min_b) * normalized_params

        if self.enforce_r1_ge_r2:
            r_pair = projected[:, 0:2]
            r_sorted, _ = torch.sort(r_pair, dim=1, descending=True)
            projected = torch.cat([r_sorted, projected[:, 2:]], dim=1)

        projected = torch.max(torch.min(projected, max_b), min_b)
        return projected

class Generator(nn.Module):
    """Generates structural parameters from a latent vector and an optional condition vector."""
    def __init__(self, latent_dim, output_dim, condition_dim):
        super().__init__()
        
        input_dim = latent_dim
        if condition_dim > 0:
            input_dim += condition_dim

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
            nn.Sigmoid()
        )

        self.projector = StructuralParamProjector()

    def forward(self, z, condition):
        if condition is not None:
            combined_input = torch.cat([z, condition], dim=1)
        else:
            combined_input = z
            
        normalized_output = self.network(combined_input)
        projected_output = self.projector(normalized_output)
        assert not torch.isnan(projected_output).any(), "Generator output contains NaN values!"
        return projected_output

if __name__ == '__main__':
    class DummyConfig:
        LATENT_DIM = 100
        STRUCT_PARAMS = ['r1', 'r2', 'w', 'g']
        METRIC_PARAMS = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
        CONDITION_DIM = 0 # Set to 0 for unconditional test
        BATCH_SIZE = 16
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        STRUCT_MIN_BOUNDS = [0.0] * len(STRUCT_PARAMS)
        STRUCT_MAX_BOUNDS = [1.0] * len(STRUCT_PARAMS)
        ENFORCE_R1_GE_R2 = False

    config = DummyConfig()

    # Test unconditional generator
    print("Unconditional Generator Test")
    unconditional_generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS),
        condition_dim=0
    ).to(config.DEVICE)
    dummy_z = torch.randn(config.BATCH_SIZE, config.LATENT_DIM).to(config.DEVICE)
    generated_params = unconditional_generator(dummy_z, None)
    print(f"Input latent vector shape: {dummy_z.shape}")
    print(f"Generated parameters shape: {generated_params.shape}")

    # Test conditional generator
    print("\nConditional Generator Test")
    config.CONDITION_DIM = len(config.METRIC_PARAMS)
    conditional_generator = Generator(
        latent_dim=config.LATENT_DIM,
        output_dim=len(config.STRUCT_PARAMS),
        condition_dim=config.CONDITION_DIM
    ).to(config.DEVICE)
    dummy_condition = torch.randn(config.BATCH_SIZE, config.CONDITION_DIM).to(config.DEVICE)
    generated_params_cond = conditional_generator(dummy_z, dummy_condition)
    print(f"Input latent vector shape: {dummy_z.shape}")
    print(f"Input condition vector shape: {dummy_condition.shape}")
    print(f"Generated parameters shape: {generated_params_cond.shape}")

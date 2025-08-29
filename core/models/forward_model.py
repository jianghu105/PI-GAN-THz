
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from config import config

class ForwardModel(nn.Module):
    """
    A simplified MLP-based forward model.
    It predicts the transmission spectrum and metrics from structural parameters.
    """
    def __init__(self, input_dim, output_dim, metrics_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.BatchNorm1d(512)
        )
        self.spectra_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, output_dim),
            nn.Sigmoid()  # Ensure output is in [0, 1] range
        )
        self.metric_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, metrics_dim)
        )

    def forward(self, struct_params):
        """
        Returns predicted_spectra and predicted_metrics.
        """
        features = self.feature_extractor(struct_params)
        predicted_spectra = self.spectra_head(features)
        predicted_metrics = self.metric_head(features)
        return predicted_spectra, predicted_metrics

if __name__ == '__main__':
    # Example usage for the simplified ForwardModel
    class DummyConfig:
        STRUCT_PARAMS = ['r1', 'r2', 'w', 'g']
        SPECTRA_PARAMS = [f'Freq_{i}' for i in range(250)]
        BATCH_SIZE = 16
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        from config import config as actual_config
    except ImportError:
        config = DummyConfig()
    else:
        config = actual_config

    model = ForwardModel(input_dim=len(config.STRUCT_PARAMS), output_dim=len(config.SPECTRA_PARAMS)).to(config.DEVICE)
    
    dummy_input = torch.randn(config.BATCH_SIZE, len(config.STRUCT_PARAMS)).to(config.DEVICE)
    
    predicted_spectra = model(dummy_input)
    
    print("Simplified Forward Model Test")
    print(f"Input shape:           {dummy_input.shape}")
    print(f"Predicted spectra shape: {predicted_spectra.shape}")


import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import config

class TrainableSpectralFeatureExtractor(nn.Module):
    """A trainable module to extract physical metrics from a spectrum."""
    def __init__(self, spectra_dim, metric_dim):
        super().__init__()
        # This network learns to extract metrics from the spectra
        self.network = nn.Sequential(
            nn.Linear(spectra_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, metric_dim)
            # No activation on the last layer, as metrics can be any real value
        )

    def forward(self, spectra_batch):
        return self.network(spectra_batch)

class ForwardModel(nn.Module):
    """Predicts the transmission spectrum from structural parameters using 1D CNN
    and extracts metrics using a trainable feature extractor."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Main network (1D CNN for spectra prediction)
        # Input: (batch_size, 1, input_dim) after unsqueeze
        self.spectra_network = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(), # Flatten the output of CNN before passing to Linear
            nn.Linear(128 * input_dim, output_dim), # input_dim is the sequence length here
            nn.Sigmoid() # Assuming spectra are normalized between 0 and 1
        )
        
        # Trainable feature extractor
        self.feature_extractor = TrainableSpectralFeatureExtractor(
            spectra_dim=output_dim,
            metric_dim=len(config.METRIC_PARAMS)
        ).to(config.DEVICE) # Ensure feature extractor is on the correct device

    def forward(self, struct_params):
        # Reshape struct_params for 1D CNN: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = struct_params.unsqueeze(1);
        predicted_spectra = self.spectra_network(x);
        
        # Pass predicted_spectra to the trainable feature extractor
        # Gradients will now flow through the feature extractor
        predicted_metrics = self.feature_extractor(predicted_spectra);
            
        return predicted_spectra, predicted_metrics

if __name__ == '__main__':
    # Example usage for ForwardModel
    # For testing, let's define dummy values if config is not fully available
    class DummyConfig:
        STRUCT_PARAMS = ['r1', 'r2', 'w', 'g']
        SPECTRA_PARAMS = [f'Freq_{i}' for i in range(250)]
        METRIC_PARAMS = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
        BATCH_SIZE = 16
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Temporarily override config for testing if not running in full project context
    try:
        from config import config as actual_config
    except ImportError:
        print("Using dummy config for testing purposes.")
        config = DummyConfig()
    else:
        config = actual_config

    model = ForwardModel(input_dim=len(config.STRUCT_PARAMS), output_dim=len(config.SPECTRA_PARAMS)).to(config.DEVICE)
    
    # Create a dummy input batch
    dummy_input = torch.randn(config.BATCH_SIZE, len(config.STRUCT_PARAMS)).to(config.DEVICE)
    
    predicted_spectra, predicted_metrics = model(dummy_input)
    
    print("Forward Model Test (1D CNN + Trainable Feature Extractor)")
    print(f"Input shape:      {dummy_input.shape}")
    print(f"Predicted spectra shape: {predicted_spectra.shape}")
    print(f"Predicted metrics shape: {predicted_metrics.shape}")
    print(f"Sample predicted metrics (first row): {predicted_metrics[0]}")

    # Test TrainableSpectralFeatureExtractor directly with a dummy spectrum
    print("\nTesting TrainableSpectralFeatureExtractor directly:")
    dummy_spectra_input = torch.randn(config.BATCH_SIZE, len(config.SPECTRA_PARAMS)).to(config.DEVICE)
    extracted_metrics = model.feature_extractor(dummy_spectra_input)
    print(f"Extracted metrics from dummy spectrum shape: {extracted_metrics.shape}")

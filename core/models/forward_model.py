import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import config

class DifferentiableSpectralFeatureExtractor(nn.Module):
    """Physics-inspired, differentiable feature extractor from spectra to metrics.

    Approximates two resonance dips using softmin attention and estimates f, Q, FoM, S.
    Metrics order: [f1, f2, Q1, FoM1, S1, Q2, FoM2, S2]
    """
    def __init__(self, spectra_dim: int, metric_dim: int):
        super().__init__()
        assert metric_dim == 8, "Metric dimension expected to be 8."
        # Frequency grid extracted from config
        freq_values = np.array([float(col.split('_')[1]) for col in config.SPECTRA_PARAMS], dtype=np.float32)
        self.register_buffer('freq_grid', torch.tensor(freq_values).view(1, -1))
        self.eps = 1e-8
        # Learnable temperatures for softmin and suppression strength
        self.log_alpha = nn.Parameter(torch.tensor(1.5))  # softmin temperature
        self.log_gamma = nn.Parameter(torch.tensor(1.0))  # suppression strength for second peak
        # Suppression width as fraction of frequency range
        self.log_sigma_frac = nn.Parameter(torch.tensor(-1.0))  # ~ exp(-1) ~ 0.37 of range

    def _softmin_weights(self, spectra: torch.Tensor) -> torch.Tensor:
        alpha = torch.nn.functional.softplus(self.log_alpha) + 1.0
        logits = -alpha * spectra
        return torch.softmax(logits, dim=1)

    def _compute_peak(self, spectra: torch.Tensor, suppress_center: torch.Tensor = None) -> tuple:
        freq = self.freq_grid.to(spectra.device)
        weights = self._softmin_weights(spectra)
        if suppress_center is not None:
            # Apply Gaussian suppression around previous peak
            fmin = torch.min(freq)
            fmax = torch.max(freq)
            frange = (fmax - fmin).clamp(min=self.eps)
            sigma = torch.nn.functional.softplus(self.log_sigma_frac) * frange
            # Compute gaussian suppression mask per batch
            # freq: (1, L), suppress_center: (B, 1)
            gauss = torch.exp(-0.5 * ((freq - suppress_center) / (sigma + self.eps)) ** 2)
            gamma = torch.nn.functional.softplus(self.log_gamma)
            # Reduce weight around previous peak
            logits = torch.log(weights + self.eps) - gamma * gauss
            weights = torch.softmax(logits, dim=1)

        # Peak frequency (soft-argmin)
        f_peak = torch.sum(weights * freq, dim=1, keepdim=True)
        # Peak amplitude
        a_peak = torch.sum(weights * spectra, dim=1, keepdim=True)
        # Approximate std as weighted std
        var = torch.sum(weights * (freq - f_peak) ** 2, dim=1, keepdim=True)
        std = torch.sqrt(var + self.eps)
        # Approximate FWHM ~ 2.355 * std for Gaussian shape
        fwhm = 2.355 * std + self.eps
        q_factor = (f_peak / fwhm).clamp(min=self.eps)
        # FoM as Q times inverse amplitude (dips -> lower better); use (1 - a) to reflect dip depth
        fom = q_factor * (1.0 + (1.0 - a_peak))
        # Sensitivity proxy as log amplitude contrast
        sensitivity = torch.log1p(1.0 + (1.0 - a_peak) * 10.0)
        return f_peak, a_peak, q_factor, fom, sensitivity

    def forward(self, spectra_batch: torch.Tensor) -> torch.Tensor:
        # spectra are expected in [0,1], where dips are near 0
        f1, a1, q1, fom1, s1 = self._compute_peak(spectra_batch)
        f2, a2, q2, fom2, s2 = self._compute_peak(spectra_batch, suppress_center=f1)
        metrics = torch.cat([f1, f2, q1, fom1, s1, q2, fom2, s2], dim=1)
        return metrics

class MetricsCalibrator(nn.Module):
    """Lightweight calibration head to align physics-inspired features with dataset-defined metrics.

    Input concatenates physics-based metrics (8 dims) with simple spectral summary stats (mean/min/max/std),
    forming a 12-dimensional vector per sample.
    """
    def __init__(self, input_dim: int = 12, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ForwardModel(nn.Module):
    """Predicts the transmission spectrum from structural parameters using MLP
    and extracts metrics using a differentiable physics-inspired feature extractor."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Main network (MLP for spectra prediction)
        self.spectra_network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        # Differentiable physics-inspired feature extractor
        self.feature_extractor = DifferentiableSpectralFeatureExtractor(
            spectra_dim=output_dim,
            metric_dim=len(config.METRIC_PARAMS)
        ).to(config.DEVICE)
        # Calibration head to increase expressivity and match dataset metric definitions
        self.calibrator = MetricsCalibrator(input_dim=12, output_dim=len(config.METRIC_PARAMS)).to(config.DEVICE)

    def forward(self, struct_params):
        spectra_logits = self.spectra_network(struct_params)
        # Activation choice via config for stability control
        if getattr(config, 'SPECTRA_ACTIVATION', 'sigmoid') == 'tanh':
            predicted_spectra = 0.5 * (torch.tanh(spectra_logits) + 1.0)
        else:
            predicted_spectra = torch.sigmoid(spectra_logits)

        # Physics-inspired baseline metrics
        physics_metrics = self.feature_extractor(predicted_spectra)
        # Simple spectral statistics as auxiliary cues
        spec_mean = torch.mean(predicted_spectra, dim=1, keepdim=True)
        spec_min, _ = torch.min(predicted_spectra, dim=1, keepdim=True)
        spec_max, _ = torch.max(predicted_spectra, dim=1, keepdim=True)
        spec_std = torch.std(predicted_spectra, dim=1, keepdim=True)
        summary_stats = torch.cat([spec_mean, spec_min, spec_max, spec_std], dim=1)
        # Calibrated metrics
        calib_input = torch.cat([physics_metrics, summary_stats], dim=1)
        predicted_metrics = self.calibrator(calib_input)
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
    
    print("Forward Model Test (MLP + Trainable Feature Extractor)")
    print(f"Input shape:      {dummy_input.shape}")
    print(f"Predicted spectra shape: {predicted_spectra.shape}")
    print(f"Predicted metrics shape: {predicted_metrics.shape}")
    print(f"Sample predicted metrics (first row): {predicted_metrics[0]}")

    # Test TrainableSpectralFeatureExtractor directly with a dummy spectrum
    print("\nTesting TrainableSpectralFeatureExtractor directly:")
    dummy_spectra_input = torch.randn(config.BATCH_SIZE, len(config.SPECTRA_PARAMS)).to(config.DEVICE)
    extracted_metrics = model.feature_extractor(dummy_spectra_input)
    print(f"Extracted metrics from dummy spectrum shape: {extracted_metrics.shape}")
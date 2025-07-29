import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from config import config

class SpectralFeatureExtractor(nn.Module):
    """A non-trainable module to extract physical metrics from a spectrum.

    This module reimplements the logic from `dataset/thz_data_processor.py`
    using PyTorch tensor operations to be GPU-compatible.
    """
    def __init__(self, freq_min=0.50, freq_max=2.99, num_freq_points=250):
        super().__init__()
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.num_freq_points = num_freq_points
        self.freq_tensor = torch.linspace(freq_min, freq_max, num_freq_points).to(config.DEVICE)
        self.half_power_db = 3.0 # For Q-factor calculation

    def _find_peaks_batch(self, spectra_batch):
        # Input spectra_batch: (batch_size, num_freq_points)
        # We are looking for dips in the transmission spectrum, so we negate it.
        neg_spectra = -spectra_batch

        batch_size, num_points = neg_spectra.shape
        all_peak_indices = torch.full((batch_size, 2), -1, dtype=torch.long, device=config.DEVICE)
        all_peak_values = torch.full((batch_size, 2), -float('inf'), dtype=torch.float32, device=config.DEVICE)

        # Apply a simple moving average for smoothing to reduce noise
        # Kernel size for smoothing (must be odd)
        kernel_size = 5
        if num_points < kernel_size: # Handle cases where spectrum is too short
            smoothed_neg_spectra = neg_spectra
        else:
            padding = kernel_size // 2
            smoothed_neg_spectra = torch.nn.functional.avg_pool1d(
                neg_spectra.unsqueeze(1), kernel_size=kernel_size, stride=1, padding=padding
            ).squeeze(1)

        for i in range(batch_size):
            current_neg_spectrum = smoothed_neg_spectra[i]
            
            # Find local maxima (dips in original spectrum)
            # Compare with left and right neighbors
            is_local_max = (current_neg_spectrum[1:-1] > current_neg_spectrum[:-2]) & \
                           (current_neg_spectrum[1:-1] > current_neg_spectrum[2:])
            
            # Adjust indices for the original spectrum (due to slicing)
            local_max_indices = torch.nonzero(is_local_max).squeeze(1) + 1 
            
            if local_max_indices.numel() == 0:
                continue

            # Filter by a simple height threshold (e.g., only consider dips below a certain value)
            # This threshold might need tuning based on data characteristics
            # For now, we'll just take the top 2 most negative points (deepest dips)
            
            # Get values of potential peaks from the ORIGINAL (unsmoothed) spectrum
            candidate_values = neg_spectra[i, local_max_indices]
            
            # Sort by value (descending) to get the most prominent dips
            sorted_values, sorted_indices_in_candidates = torch.sort(candidate_values, descending=True)
            
            # Get the original indices of the top 2 peaks
            top_2_original_indices = local_max_indices[sorted_indices_in_candidates[:min(2, len(sorted_values))]]
            
            # Ensure indices are sorted by frequency (ascending)
            top_2_original_indices = torch.sort(top_2_original_indices).values
            
            for j, idx in enumerate(top_2_original_indices):
                if j < 2: # Only take up to 2 peaks
                    all_peak_indices[i, j] = idx
                    all_peak_values[i, j] = neg_spectra[i, idx] # Store actual negative magnitude at peaks

        return all_peak_indices, all_peak_values

    def _interpolate_frequency(self, f1, f2, m1, m2, target_m):
        # Linear interpolation to find frequency at target magnitude
        # Handles cases where m1 == m2 to avoid division by zero
        # Returns f1 if m1 == m2, assuming it's a flat region
        return torch.where(
            (m2 - m1) == 0,
            f1,
            f1 + (f2 - f1) * (target_m - m1) / (m2 - m1)
        )

    def _calculate_q_factor_batch(self, freq_batch, mag_batch, peak_indices):
        # freq_batch: (num_freq_points,)
        # mag_batch: (batch_size, num_freq_points)
        # peak_indices: (batch_size,) - index of the peak for each sample

        batch_size = mag_batch.shape[0]
        q_factors = torch.full((batch_size,), float('nan'), device=config.DEVICE)

        # Create a batch of frequency tensors for vectorized operations
        freq_batch_expanded = freq_batch.unsqueeze(0).expand(batch_size, -1)

        for i in range(batch_size):
            peak_idx = peak_indices[i]
            if peak_idx == -1: # No valid peak
                continue

            peak_freq = freq_batch[peak_idx]
            peak_mag = mag_batch[i, peak_idx]
            half_power_target = peak_mag + self.half_power_db

            # Find points where magnitude crosses the half_power_target
            # We look for the first point to the left and right where mag < half_power_target
            
            # Left side search
            left_cross_indices = torch.nonzero(mag_batch[i, :peak_idx] < half_power_target).squeeze(1)
            if left_cross_indices.numel() > 0:
                idx1_left = left_cross_indices[-1] # Last index where mag < target
                idx2_left = idx1_left + 1 # First index where mag >= target (or peak_idx)
                
                if idx2_left <= peak_idx: # Ensure idx2_left is not beyond peak_idx
                    f_left = self._interpolate_frequency(
                        freq_batch[idx1_left], freq_batch[idx2_left],
                        mag_batch[i, idx1_left], mag_batch[i, idx2_left],
                        half_power_target
                    )
                else:
                    f_left = freq_batch[peak_idx] # Fallback to peak freq if interpolation fails
            else:
                f_left = freq_batch[0] # If no point to the left is below target, use first freq

            # Right side search
            right_cross_indices = torch.nonzero(mag_batch[i, peak_idx:] < half_power_target).squeeze(1) + peak_idx
            if right_cross_indices.numel() > 0:
                idx1_right = right_cross_indices[0] # First index where mag < target
                idx2_right = idx1_right - 1 # Last index where mag >= target (or peak_idx)

                if idx2_right >= peak_idx: # Ensure idx2_right is not before peak_idx
                    f_right = self._interpolate_frequency(
                        freq_batch[idx2_right], freq_batch[idx1_right],
                        mag_batch[i, idx2_right], mag_batch[i, idx1_right],
                        half_power_target
                    )
                else:
                    f_right = freq_batch[peak_idx] # Fallback to peak freq if interpolation fails
            else:
                f_right = freq_batch[-1] # If no point to the right is below target, use last freq

            fwhm = f_right - f_left
            
            # Ensure FWHM is positive and finite
            if fwhm > 0 and torch.isfinite(fwhm) and torch.isfinite(peak_freq):
                q_factors[i] = peak_freq / fwhm
            else:
                q_factors[i] = 0.0 # Assign 0.0 instead of NaN for invalid FWHM

        return q_factors

    def _calculate_fom_batch(self, q_factors, magnitudes):
        # magnitudes are in dB, convert to linear scale
        linear_magnitudes = torch.pow(10, torch.abs(magnitudes) / 20.0)
        fom = q_factors * linear_magnitudes
        fom[~torch.isfinite(fom)] = 0.0 # Handle NaNs from Q-factors by setting to 0
        return fom

    def _calculate_sensitivity_batch(self, magnitudes):
        # magnitudes are in dB, convert to linear scale
        linear_magnitudes = torch.pow(10, torch.abs(magnitudes) / 20.0)
        # Sensitivity defined as log10 of linear magnitude * 10, ensure positive
        sensitivity = torch.log10(linear_magnitudes + 1e-9) * 10 # Add small epsilon to avoid log(0)
        sensitivity = torch.max(torch.zeros_like(sensitivity), sensitivity)
        sensitivity[~torch.isfinite(sensitivity)] = 0.0 # Handle NaNs by setting to 0
        return sensitivity

    def forward(self, spectra_batch):
        # spectra_batch is (batch_size, num_freq_points)
        batch_size = spectra_batch.shape[0]

        # Initialize metrics tensor with NaNs
        metrics = torch.full((batch_size, len(config.METRIC_PARAMS)), float('nan'), device=config.DEVICE)

        # Find the two most prominent peaks (dips)
        peak_indices_batch, peak_values_batch = self._find_peaks_batch(spectra_batch)
        
        # Extract f1, f2 (frequencies at peaks)
        f1_indices = peak_indices_batch[:, 0]
        f2_indices = peak_indices_batch[:, 1]

        # Handle cases where no valid peak was found (-1 index)
        f1 = torch.where(f1_indices != -1, self.freq_tensor[f1_indices], torch.tensor(float('nan'), device=config.DEVICE))
        f2 = torch.where(f2_indices != -1, self.freq_tensor[f2_indices], torch.tensor(float('nan'), device=config.DEVICE))

        # Calculate Q1, Q2
        q1 = self._calculate_q_factor_batch(self.freq_tensor, spectra_batch, f1_indices)
        q2 = self._calculate_q_factor_batch(self.freq_tensor, spectra_batch, f2_indices)

        # Calculate FoM1, FoM2
        # Need the actual magnitude at the peak for FoM calculation
        mag_at_f1 = torch.where(f1_indices != -1, spectra_batch[torch.arange(batch_size), f1_indices], torch.tensor(float('nan'), device=config.DEVICE))
        mag_at_f2 = torch.where(f2_indices != -1, spectra_batch[torch.arange(batch_size), f2_indices], torch.tensor(float('nan'), device=config.DEVICE))

        fom1 = self._calculate_fom_batch(q1, mag_at_f1)
        fom2 = self._calculate_fom_batch(q2, mag_at_f2)

        # Calculate S1, S2
        s1 = self._calculate_sensitivity_batch(mag_at_f1)
        s2 = self._calculate_sensitivity_batch(mag_at_f2)

        # Assign to metrics tensor based on config.METRIC_PARAMS order
        # Ensure the order matches config.METRIC_PARAMS: ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
        metrics[:, 0] = f1
        metrics[:, 1] = f2
        metrics[:, 2] = q1
        metrics[:, 3] = fom1
        metrics[:, 4] = s1
        metrics[:, 5] = q2
        metrics[:, 6] = fom2
        metrics[:, 7] = s2
        
        # Replace NaNs with 0 or a small constant if they cause issues in downstream tasks
        # For now, keep NaNs to indicate invalid calculations, but be aware for loss functions.
        # A common practice is to replace NaNs with 0 or a very small number for loss calculation
        # or to mask them out. For now, we'll let the loss function handle NaNs.
        
        return metrics

class ForwardModel(nn.Module):
    """Predicts the transmission spectrum from structural parameters."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_dim),
            nn.Sigmoid() # Assuming spectra are normalized between 0 and 1
        )
        self.feature_extractor = SpectralFeatureExtractor(
            freq_min=0.50, freq_max=2.99, num_freq_points=output_dim
        ).to(config.DEVICE) # Ensure feature extractor is on the correct device

    def forward(self, struct_params):
        predicted_spectra = self.network(struct_params)
        
        # The feature extractor is not trained, so we detach the gradients
        # and ensure it's in eval mode if it had any trainable components (it doesn't here)
        with torch.no_grad():
            predicted_metrics = self.feature_extractor(predicted_spectra)
            
        return predicted_spectra, predicted_metrics

if __name__ == '__main__':
    # Example usage for ForwardModel
    model = ForwardModel(input_dim=len(config.STRUCT_PARAMS), output_dim=len(config.SPECTRA_PARAMS)).to(config.DEVICE)
    
    # Create a dummy input batch
    dummy_input = torch.randn(config.BATCH_SIZE, len(config.STRUCT_PARAMS)).to(config.DEVICE)
    
    predicted_spectra, predicted_metrics = model(dummy_input)
    
    print("Forward Model Test")
    print(f"Input shape:      {dummy_input.shape}")
    print(f"Predicted spectra shape: {predicted_spectra.shape}")
    print(f"Predicted metrics shape: {predicted_metrics.shape}")
    print(f"Sample predicted metrics (first row): {predicted_metrics[0]}")

    # Test SpectralFeatureExtractor directly with a dummy spectrum
    print("\nTesting SpectralFeatureExtractor directly:")
    # Create a dummy spectrum that might have some dips
    # Example: a simple V-shape dip
    test_spectra = torch.ones(1, len(config.SPECTRA_PARAMS), device=config.DEVICE) * 0.5
    # Create a dip around index 50 and another around index 150
    test_spectra[0, 45:55] = torch.linspace(0.5, 0.1, 10) # First dip
    test_spectra[0, 145:155] = torch.linspace(0.5, 0.1, 10) # Second dip

    # Normalize to [0,1] if the model outputs sigmoid
    # If the original data is in dB, this needs to be handled carefully.
    # Assuming the model outputs normalized spectra, and feature extractor expects that.
    
    extracted_metrics = model.feature_extractor(test_spectra)
    print(f"Extracted metrics from dummy spectrum: {extracted_metrics[0]}")
    # Expected output for f1, f2, Q1, FoM1, S1, Q2, FoM2, S2
    # f1 and f2 should be around the frequencies corresponding to indices 45-55 and 145-155
    # Q, FoM, S will depend on the exact shape of the dip.
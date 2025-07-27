import torch
import argparse
import os
import sys
import numpy as np
from typing import Dict, Any
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.models.forward_model import ForwardModel, EnhancedForwardPINN
from core.utils.data_loader import MetamaterialDataset, denormalize_metrics
import config.config as cfg

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    计算回归评估指标
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        
    Returns:
        Dict[str, float]: 评估指标字典
    """
    metrics = {}
    
    # 基础回归指标
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # 相关性指标
    try:
        metrics['r2'] = r2_score(y_true, y_pred)
    except:
        metrics['r2'] = float('nan')
        
    # 相关系数
    try:
        if y_true.ndim == 1:
            pearson_corr, _ = pearsonr(y_true, y_pred)
            metrics['pearson_r'] = pearson_corr
        else:
            # 多维数据计算平均相关系数
            pearson_corrs = []
            for i in range(y_true.shape[1]):
                try:
                    p_corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
                    pearson_corrs.append(p_corr)
                except:
                    pass
                    
            metrics['pearson_r'] = np.mean(pearson_corrs) if pearson_corrs else float('nan')
    except:
        metrics['pearson_r'] = float('nan')
    
    # 相对误差
    metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    return metrics

def evaluate_forward_model(config_path: str = None, num_samples: int = 1000) -> Dict[str, Any]:
    """
    Evaluates the pre-trained forward model.
    """
    print(f"\n=== Forward Model Evaluation ({num_samples} samples) ===")

    # Load configuration
    config = Config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    # Assuming EnhancedForwardPINN is the primary forward model used
    model = EnhancedForwardPINN(
        input_param_dim=config.model.forward_model.input_dim,
        spectrum_dim=config.model.forward_model.output_dim
    ).to(device)
    
    # Load pre-trained weights
    if config.evaluation.forward_model.pretrained_path:
        try:
            model.load_state_dict(torch.load(config.evaluation.forward_model.pretrained_path, map_location=device))
            print(f"Loaded pre-trained forward model from {config.evaluation.forward_model.pretrained_path}")
        except FileNotFoundError:
            print(f"Error: Pre-trained model not found at {config.evaluation.forward_model.pretrained_path}")
            return {}
    else:
        print("Warning: No pre-trained path specified for forward model evaluation.")
        return {}

    model.eval() # Set model to evaluation mode

    # Prepare data loader for evaluation
    try:
        dataset = MetamaterialDataset(
            data_path=config.data.eval_data_path, # Assuming an eval_data_path in config
            num_points_per_sample=config.model.forward_model.output_dim # Spectrum dim
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = Subset(dataset, sample_indices)
    eval_data_loader = DataLoader(subset, batch_size=config.evaluation.batch_size, shuffle=False)

    all_real_spectra = []
    all_pred_spectra = []
    all_real_metrics = []
    all_pred_metrics = []

    with torch.no_grad(): # Disable gradient calculation for evaluation
        for batch_idx, (real_spectrum, _, real_params_norm, real_metrics_denorm, real_metrics_norm) in enumerate(eval_data_loader):
            real_spectrum = real_spectrum.to(device)
            real_params_norm = real_params_norm.to(device)
            real_metrics_norm = real_metrics_norm.to(device)

            pred_spectrum, pred_metrics_norm = model(real_params_norm)
            pred_metrics_denorm = denormalize_metrics(pred_metrics_norm, dataset.metric_ranges)

            # Collect results
            all_real_spectra.append(real_spectrum.cpu().numpy())
            all_pred_spectra.append(pred_spectrum.cpu().numpy())
            all_real_metrics.append(real_metrics_denorm.cpu().numpy())
            all_pred_metrics.append(pred_metrics_denorm.cpu().numpy())

    # Concatenate results
    all_real_spectra = np.concatenate(all_real_spectra, axis=0)
    all_pred_spectra = np.concatenate(all_pred_spectra, axis=0)
    all_real_metrics = np.concatenate(all_real_metrics, axis=0)
    all_pred_metrics = np.concatenate(all_pred_metrics, axis=0)

    # Calculate evaluation metrics
    spectrum_metrics = calculate_metrics(all_real_spectra, all_pred_spectra)
    metrics_metrics = calculate_metrics(all_real_metrics, all_pred_metrics)

    results = {
        'spectrum_prediction': spectrum_metrics,
        'metrics_prediction': metrics_metrics,
        'num_samples': len(all_real_spectra),
        'data_samples': {
            'real_spectra': all_real_spectra[:50],  # Save first 50 samples for visualization
            'pred_spectra': all_pred_spectra[:50],
            'real_metrics': all_real_metrics[:50],
            'pred_metrics': all_pred_metrics[:50]
        }
    }
    
    print(f"✓ Forward network evaluation completed")
    print(f"  - Spectrum R²: {spectrum_metrics['r2']:.4f}")
    print(f"  - Metrics R²: {metrics_metrics['r2']:.4f}")
    print("-" * 50)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the Forward Model.")
    parser.add_argument('--config', type=str, default='config/config.py',
                        help='Path to the configuration file.')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation.')
    args = parser.parse_args()
    
    evaluate_forward_model(config_path=args.config, num_samples=args.num_samples)
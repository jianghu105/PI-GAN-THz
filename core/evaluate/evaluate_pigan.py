import sys
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time
from typing import Dict, Any
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

import config.config as cfg
from core.utils.set_seed import set_seed
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics

# Import models
from core.models.generator import EnhancedGenerator, Generator
from core.models.discriminator import EnhancedDiscriminator, Discriminator
from core.models.forward_model import EnhancedForwardPINN, ForwardModel

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

def _load_pigan_models(model_dir: str, device: torch.device, enhanced: bool = False):
    """
    Helper function to load PI-GAN models.
    """
    if model_dir is None:
        model_dir = cfg.SAVED_MODELS_DIR
            
    print(f"Loading {'enhanced' if enhanced else 'standard'} models from: {model_dir}")
        
    try:
        if enhanced:
            generator = EnhancedGenerator(
                spectrum_dim=cfg.SPECTRUM_DIM,
                z_dim=cfg.Z_DIM,
                output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM
            ).to(device)
            
            discriminator = EnhancedDiscriminator(
                spectrum_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM,
                param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM
            ).to(device)
            
            forward_model = EnhancedForwardPINN(
                input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM
            ).to(device)
            
            gen_path = os.path.join(model_dir, "generator_final.pth")
            disc_path = os.path.join(model_dir, "discriminator_final.pth")
            fwd_path = os.path.join(model_dir, "forward_model_final.pth")
        else:
            generator = Generator(
                spectrum_dim=cfg.SPECTRUM_DIM,
                z_dim=cfg.Z_DIM,
                output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM
            ).to(device)
            
            discriminator = Discriminator(
                spectrum_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM,
                param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM
            ).to(device)
            
            forward_model = ForwardModel(
                input_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                output_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                hidden_dims=[256, 256] # Assuming default hidden dims for standard forward model
            ).to(device)
            
            gen_path = os.path.join(model_dir, "generator_final.pth") # Adjust if standard models have different names
            disc_path = os.path.join(model_dir, "discriminator_final.pth")
            fwd_path = os.path.join(model_dir, "forward_model_pretrained.pth") # Assuming this is the standard one

        generator.load_state_dict(torch.load(gen_path, map_location=device))
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))
        forward_model.load_state_dict(torch.load(fwd_path, map_location=device))
        
        generator.eval()
        discriminator.eval()
        forward_model.eval()
        
        print("✓ Models loaded successfully!")
        return generator, discriminator, forward_model
            
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        return None, None, None

def evaluate_pigan(model_dir: str = None, data_path: str = None, num_samples: int = 1000) -> Dict[str, Any]:
    """
    评估PI-GAN性能（生成器和判别器）
    
    Args:
        model_dir: 模型目录路径
        data_path: 数据集路径
        num_samples: 评估样本数
            
    Returns:
        Dict[str, Any]: PI-GAN评估结果
    """
    print(f"\n=== PI-GAN Evaluation ({num_samples} samples) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator, forward_model = _load_pigan_models(model_dir, device, enhanced=False)
    
    if generator is None or discriminator is None or forward_model is None:
        print("Failed to load PI-GAN models. Aborting evaluation.")
        return {}

    try:
        dataset = MetamaterialDataset(
            data_path=data_path if data_path else cfg.DATASET_PATH,
            num_points_per_sample=cfg.SPECTRUM_DIM
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = Subset(dataset, sample_indices)
    dataloader = DataLoader(subset, batch_size=64, shuffle=False)
    
    all_real_params = []
    all_pred_params = []
    all_real_scores = []
    all_fake_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
            
            real_spectrum = real_spectrum.to(device)
            real_params_denorm = real_params_denorm.to(device)
            real_params_norm = real_params_norm.to(device)
            
            # 生成器预测参数
            latent_vector = torch.randn(real_spectrum.shape[0], cfg.Z_DIM).to(device)
            if isinstance(generator, EnhancedGenerator):
                pred_params_norm = generator(real_spectrum, latent_vector)
            else:
                pred_params_norm = generator(real_spectrum)
            pred_params_denorm = denormalize_params(pred_params_norm, dataset.param_ranges)
            
            # 判别器评分
            if isinstance(discriminator, EnhancedDiscriminator):
                real_scores, _ = discriminator(real_spectrum, real_params_denorm)
                fake_scores, _ = discriminator(real_spectrum, pred_params_denorm)
            else:
                real_scores = discriminator(real_spectrum, real_params_denorm)
                fake_scores = discriminator(real_spectrum, pred_params_denorm)
            
            # 收集结果
            all_real_params.append(real_params_denorm.cpu().numpy())
            all_pred_params.append(pred_params_denorm.cpu().numpy())
            all_real_scores.append(real_scores.cpu().numpy())
            all_fake_scores.append(fake_scores.cpu().numpy())
    
    # 合并结果
    all_real_params = np.concatenate(all_real_params, axis=0)
    all_pred_params = np.concatenate(all_pred_params, axis=0)
    all_real_scores = np.concatenate(all_real_scores, axis=0)
    all_fake_scores = np.concatenate(all_fake_scores, axis=0)
    
    # 计算评估指标
    param_metrics = calculate_metrics(all_real_params, all_pred_params)
    
    # 判别器性能
    real_accuracy = np.mean(all_real_scores > 0.5)
    fake_accuracy = np.mean(all_fake_scores < 0.5)
    overall_accuracy = (real_accuracy + fake_accuracy) / 2
    
    results = {
        'parameter_prediction': param_metrics,
        'discriminator_performance': {
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'overall_accuracy': overall_accuracy,
            'real_score_mean': np.mean(all_real_scores),
            'fake_score_mean': np.mean(all_fake_scores)
        },
        'num_samples': len(all_real_params),
        'data_samples': {
            'real_params': all_real_params[:50],  # 保存前50个样本用于可视化
            'pred_params': all_pred_params[:50]
        },
        'score_distributions': {
            'real_scores': all_real_scores[:200],  # 保存前200个得分用于可视化
            'fake_scores': all_fake_scores[:200]
        }
    }
    
    print(f"✓ PI-GAN evaluation completed")
    print(f"  - Parameter R²: {param_metrics['r2']:.4f}")
    print(f"  - Discriminator Accuracy: {overall_accuracy:.4f}")
    print("-" * 50)
    
    return results

def evaluate_pigan_enhanced(model_dir: str = None, data_path: str = None, num_samples: int = 1000) -> Dict[str, Any]:
    """
    评估增强版PI-GAN性能（生成器、判别器和物理一致性）
    
    Args:
        model_dir: 模型目录路径
        data_path: 数据集路径
        num_samples: 评估样本数
            
    Returns:
        Dict[str, Any]: 增强版PI-GAN评估结果
    """
    print(f"\n=== Enhanced PI-GAN Evaluation ({num_samples} samples) ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator, discriminator, forward_model = _load_pigan_models(model_dir, device, enhanced=True)
    
    if generator is None or discriminator is None or forward_model is None:
        print("Failed to load enhanced PI-GAN models. Aborting evaluation.")
        return {}

    try:
        dataset = MetamaterialDataset(
            data_path=data_path if data_path else cfg.DATASET_PATH,
            num_points_per_sample=cfg.SPECTRUM_DIM
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

    # 确认使用的是增强模型
    if not (isinstance(generator, EnhancedGenerator) and 
            isinstance(discriminator, EnhancedDiscriminator)):
        print("Warning: Models are not enhanced versions. Using standard evaluation.")
        return evaluate_pigan(model_dir, data_path, num_samples)
        
    sample_indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    subset = Subset(dataset, sample_indices)
    dataloader = DataLoader(subset, batch_size=64, shuffle=False)
    
    all_real_params = []
    all_pred_params = []
    all_real_scores = []
    all_fake_scores = []
    all_physics_scores = []
    
    with torch.no_grad():
        for batch in dataloader:
            real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
            
            real_spectrum = real_spectrum.to(device)
            real_params_denorm = real_params_denorm.to(device)
            real_params_norm = real_params_norm.to(device)
            
            # 生成器预测参数
            latent_vector = torch.randn(real_spectrum.shape[0], cfg.Z_DIM).to(device)
            pred_params_norm = generator(real_spectrum, latent_vector)
            pred_params_denorm = denormalize_params(pred_params_norm, dataset.param_ranges)
            
            # 判别器评分
            real_scores_rf, real_scores_physics = discriminator(real_spectrum, real_params_denorm)
            fake_scores_rf, fake_scores_physics = discriminator(real_spectrum, pred_params_denorm)
            
            # 收集结果
            all_real_params.append(real_params_denorm.cpu().numpy())
            all_pred_params.append(pred_params_denorm.cpu().numpy())
            all_real_scores.append(real_scores_rf.cpu().numpy())
            all_fake_scores.append(fake_scores_rf.cpu().numpy())
            all_physics_scores.append(fake_scores_physics.cpu().numpy())
    
    # 合并结果
    all_real_params = np.concatenate(all_real_params, axis=0)
    all_pred_params = np.concatenate(all_pred_params, axis=0)
    all_real_scores = np.concatenate(all_real_scores, axis=0)
    all_fake_scores = np.concatenate(all_fake_scores, axis=0)
    all_physics_scores = np.concatenate(all_physics_scores, axis=0)
    
    # 计算评估指标
    param_metrics = calculate_metrics(all_real_params, all_pred_params)
    
    # 判别器性能
    real_accuracy = np.mean(all_real_scores > 0.5)
    fake_accuracy = np.mean(all_fake_scores < 0.5)
    overall_accuracy = (real_accuracy + fake_accuracy) / 2
    
    # 物理一致性性能
    physics_score_mean = np.mean(all_physics_scores)
    physics_score_std = np.std(all_physics_scores)
    
    results = {
        'parameter_prediction': param_metrics,
        'discriminator_performance': {
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'overall_accuracy': overall_accuracy,
            'real_score_mean': np.mean(all_real_scores),
            'fake_score_mean': np.mean(all_fake_scores)
        },
        'physics_consistency': {
            'physics_score_mean': physics_score_mean,
            'physics_score_std': physics_score_std
        },
        'num_samples': len(all_real_params),
        'data_samples': {
            'real_params': all_real_params[:50],  # 保存前50个样本用于可视化
            'pred_params': all_pred_params[:50]
        },
        'score_distributions': {
            'real_scores': all_real_scores[:200],  # 保存前200个得分用于可视化
            'fake_scores': all_fake_scores[:200],
            'physics_scores': all_physics_scores[:200]
        }
    }
    
    print(f"✓ Enhanced PI-GAN evaluation completed")
    print(f"  - Parameter R²: {param_metrics['r2']:.4f}")
    print(f"  - Discriminator Accuracy: {overall_accuracy:.4f}")
    print(f"  - Physics Consistency Score: {physics_score_mean:.4f}")
    print("-" * 50)
    
    return results

def plot_pigan_evaluation_enhanced(results: Dict[str, Any], save_dir: str = None):
    """Generate plots for enhanced PI-GAN evaluation"""
    
    # Set up English plotting
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Enhanced PI-GAN Model Evaluation Results', fontsize=16)
    
    param_metrics = results['parameter_prediction']
    disc_metrics = results['discriminator_performance']
    physics_metrics = results['physics_consistency']
    
    # 1. Generator Performance (R² and Error Metrics)
    gen_categories = ['R²', 'MAE', 'RMSE', 'Pearson R', 'MAPE (%)']
    gen_values = [
        param_metrics['r2'],
        param_metrics['mae'],
        param_metrics['rmse'],
        param_metrics['pearson_r'] if not np.isnan(param_metrics['pearson_r']) else 0,
        param_metrics['mape']
    ]
    
    colors = ['green' if gen_values[0] > 0.8 else 'orange' if gen_values[0] > 0.6 else 'red'] + ['blue'] * 4
    bars = axes[0, 0].bar(gen_categories, gen_values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Generator Performance Metrics')
    axes[0, 0].set_ylabel('Metric Value')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, gen_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + max(gen_values) * 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Discriminator Performance
    disc_categories = ['Real\nAccuracy', 'Fake\nAccuracy', 'Overall\nAccuracy']
    disc_values = [
        disc_metrics['real_accuracy'],
        disc_metrics['fake_accuracy'],
        disc_metrics['overall_accuracy']
    ]
    
    colors_disc = ['green' if val > 0.8 else 'orange' if val > 0.6 else 'red' for val in disc_values]
    bars_disc = axes[0, 1].bar(disc_categories, disc_values, color=colors_disc, alpha=0.7)
    axes[0, 1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Target (0.8)')
    axes[0, 1].set_title('Discriminator Accuracy')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    
    # Add value labels
    for bar, value in zip(bars_disc, disc_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Physics Consistency
    axes[0, 2].bar(['Physics\nScore'], [physics_metrics['physics_score_mean']], 
                   color='purple', alpha=0.7)
    axes[0, 2].errorbar(['Physics\nScore'], [physics_metrics['physics_score_mean']], 
                        yerr=[physics_metrics['physics_score_std']], 
                        fmt='o', color='black', capsize=5)
    axes[0, 2].axhline(y=0.3, color='green', linestyle='--', alpha=0.7, label='Target (<0.3)')
    axes[0, 2].set_title('Physics Consistency')
    axes[0, 2].set_ylabel('Physics Score')
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].legend()
    
    # Add value labels
    axes[0, 2].text(0, physics_metrics['physics_score_mean'] + 0.02,
                   f"{physics_metrics['physics_score_mean']:.3f}±{physics_metrics['physics_score_std']:.3f}", 
                   ha='center', va='bottom', fontweight='bold')
    
    # 4. Parameter Prediction Scatter Plot (if data available)
    if 'data_samples' in results and 'real_params' in results['data_samples']:
        data_samples = results['data_samples']
        real_params = data_samples['real_params']
        pred_params = data_samples['pred_params']
        
        # Select first parameter for visualization
        axes[1, 0].scatter(real_params[:, 0], pred_params[:, 0], alpha=0.6, color='blue')
        
        # Add perfect prediction line
        min_val = min(real_params[:, 0].min(), pred_params[:, 0].min())
        max_val = max(real_params[:, 0].max(), pred_params[:, 0].max())
        axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
        
        axes[1, 0].set_xlabel('Real Parameter Values')
        axes[1, 0].set_ylabel('Predicted Parameter Values')
        axes[1, 0].set_title('Parameter Prediction Accuracy (First Parameter)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Parameter prediction\nscatter plot not available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, alpha=0.7)
        axes[1, 0].set_title('Parameter Prediction Scatter Plot')
    
    # 5. Score Histograms (if data available)
    if 'score_distributions' in results:
        score_data = results['score_distributions']
        if 'real_scores' in score_data and 'fake_scores' in score_data:
            axes[1, 1].hist(score_data['real_scores'], bins=20, alpha=0.7, 
                           label='Real Scores', color='blue', density=True)
            axes[1, 1].hist(score_data['fake_scores'], bins=20, alpha=0.7, 
                           label='Fake Scores', color='red', density=True)
            axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision Boundary')
            axes[1, 1].set_xlabel('Discriminator Score')
            axes[1, 1].set_ylabel('Density')
            axes[1, 1].set_title('Score Distribution Comparison')
            axes[1, 1].legend()
        else:
            axes[1, 1].text(0.5, 0.5, 'Score distribution\ndata not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, alpha=0.7)
            axes[1, 1].set_title('Score Distribution Comparison')
    else:
        axes[1, 1].text(0.5, 0.5, 'Score distribution\ndata not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, alpha=0.7)
        axes[1, 1].set_title('Score Distribution Comparison')
    
    # 6. Performance Summary Radar Chart
    ax_radar = plt.subplot(2, 3, 6, projection='polar')
    
    categories_radar = ['Generator\nR²', 'Discriminator\nAccuracy', 'Physics\nConsistency']
    values_radar = [
        param_metrics['r2'],
        disc_metrics['overall_accuracy'],
        1 - physics_metrics['physics_score_mean']  # Inverted for better visualization
    ]
    
    angles = np.linspace(0, 2*np.pi, len(categories_radar), endpoint=False).tolist()
    values_radar += values_radar[:1]
    angles += angles[:1]
    
    ax_radar.plot(angles, values_radar, 'o-', linewidth=2, label='PI-GAN Performance', color='purple')
    ax_radar.fill(angles, values_radar, alpha=0.25, color='purple')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories_radar)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Enhanced PI-GAN Overall Performance Radar')
    
    plt.tight_layout()
    
    # Save plot
    if save_dir is None:
        save_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(save_dir, f"pigan_enhanced_evaluation_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Enhanced PI-GAN evaluation plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PI-GAN model performance")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    parser.add_argument('--enhanced', action='store_true',
                        help='Whether to evaluate enhanced PI-GAN')
    
    args = parser.parse_args()
    
    set_seed(cfg.RANDOM_SEED)

    if args.enhanced:
        evaluate_pigan_enhanced(
            model_dir=args.model_dir,
            data_path=args.data_path,
            num_samples=args.num_samples
        )
    else:
        evaluate_pigan(
            model_dir=args.model_dir,
            data_path=args.data_path,
            num_samples=args.num_samples
        )

if __name__ == "__main__":
    main()
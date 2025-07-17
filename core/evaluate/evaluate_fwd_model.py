# PI_GAN_THZ/core/evaluate/evaluate_fwd_model.py

import sys
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入统一评估器
from core.evaluate.unified_evaluator import UnifiedEvaluator
import config.config as cfg
from core.utils.set_seed import set_seed

def evaluate_forward_model(model_dir: str = None, 
                          data_path: str = None, 
                          num_samples: int = 1000):
    """
    评估前向模型性能
    
    Args:
        model_dir: 模型目录路径
        data_path: 数据集路径
        num_samples: 评估样本数
    """
    print("\n--- Forward Model Evaluation ---")
    
    # 设置随机种子
    set_seed(cfg.RANDOM_SEED)
    
    # 创建评估器
    evaluator = UnifiedEvaluator()
    
    # 加载模型和数据
    if not evaluator.load_models(model_dir):
        print("❌ Failed to load models!")
        return
    
    if not evaluator.load_dataset(data_path):
        print("❌ Failed to load dataset!")
        return
    
    # 运行前向网络评估
    results = evaluator.evaluate_forward_network(num_samples)
    
    # 打印详细结果
    print("\n📊 Forward Model Evaluation Results:")
    print("-" * 50)
    
    spectrum_metrics = results['spectrum_prediction']
    print("Spectrum Prediction:")
    print(f"  - R²: {spectrum_metrics['r2']:.6f}")
    print(f"  - MAE: {spectrum_metrics['mae']:.6f}")
    print(f"  - RMSE: {spectrum_metrics['rmse']:.6f}")
    print(f"  - Pearson R: {spectrum_metrics['pearson_r']:.6f}")
    print(f"  - MAPE: {spectrum_metrics['mape']:.2f}%")
    
    metrics_metrics = results['metrics_prediction']
    print("\nMetrics Prediction:")
    print(f"  - R²: {metrics_metrics['r2']:.6f}")
    print(f"  - MAE: {metrics_metrics['mae']:.6f}")
    print(f"  - RMSE: {metrics_metrics['rmse']:.6f}")
    print(f"  - Pearson R: {metrics_metrics['pearson_r']:.6f}")
    print(f"  - MAPE: {metrics_metrics['mape']:.2f}%")
    
    # 性能评估
    print("\n🎯 Performance Assessment:")
    if spectrum_metrics['r2'] > 0.9 and metrics_metrics['r2'] > 0.9:
        print("✅ Forward model shows EXCELLENT performance!")
    elif spectrum_metrics['r2'] > 0.8 and metrics_metrics['r2'] > 0.8:
        print("✅ Forward model shows GOOD performance!")
    elif spectrum_metrics['r2'] > 0.6 and metrics_metrics['r2'] > 0.6:
        print("⚠️ Forward model shows MODERATE performance.")
    else:
        print("❌ Forward model shows POOR performance and needs improvement.")
    
    # Generate evaluation plots
    plot_forward_model_evaluation(results)
    
    print(f"\n✅ Forward model evaluation completed with {results['num_samples']} samples.")

def plot_forward_model_evaluation(results):
    """Generate plots for forward model evaluation"""
    
    # Set up English plotting
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Forward Model Evaluation Results', fontsize=16)
    
    spectrum_metrics = results['spectrum_prediction']
    metrics_metrics = results['metrics_prediction']
    
    # 1. R² Score Comparison
    categories = ['Spectrum\nPrediction', 'Metrics\nPrediction']
    r2_scores = [spectrum_metrics['r2'], metrics_metrics['r2']]
    colors = ['blue' if r2 > 0.8 else 'orange' if r2 > 0.6 else 'red' for r2 in r2_scores]
    
    bars = axes[0, 0].bar(categories, r2_scores, color=colors, alpha=0.7)
    axes[0, 0].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Target (0.8)')
    axes[0, 0].set_title('R² Score Comparison')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()
    
    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Error Metrics Comparison
    error_types = ['MAE', 'RMSE', 'MAPE']
    spectrum_errors = [spectrum_metrics['mae'], spectrum_metrics['rmse'], spectrum_metrics['mape']/100]
    metrics_errors = [metrics_metrics['mae'], metrics_metrics['rmse'], metrics_metrics['mape']/100]
    
    x = np.arange(len(error_types))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, spectrum_errors, width, label='Spectrum', alpha=0.7, color='blue')
    axes[0, 1].bar(x + width/2, metrics_errors, width, label='Metrics', alpha=0.7, color='orange')
    axes[0, 1].set_title('Error Metrics Comparison')
    axes[0, 1].set_ylabel('Error Value')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(error_types)
    axes[0, 1].legend()
    axes[0, 1].set_yscale('log')
    
    # 3. Performance Radar Chart
    ax_radar = plt.subplot(2, 2, 3, projection='polar')
    
    categories_radar = ['R²', 'Pearson\nCorrelation', 'Low MAE\n(1-MAE)', 'Low RMSE\n(1-RMSE)']
    spectrum_values = [
        spectrum_metrics['r2'],
        spectrum_metrics['pearson_r'] if not np.isnan(spectrum_metrics['pearson_r']) else 0,
        1 - min(spectrum_metrics['mae'], 1.0),
        1 - min(spectrum_metrics['rmse'], 1.0)
    ]
    
    angles = np.linspace(0, 2*np.pi, len(categories_radar), endpoint=False).tolist()
    spectrum_values += spectrum_values[:1]
    angles += angles[:1]
    
    ax_radar.plot(angles, spectrum_values, 'o-', linewidth=2, label='Spectrum Prediction', color='blue')
    ax_radar.fill(angles, spectrum_values, alpha=0.25, color='blue')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories_radar)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Spectrum Prediction Performance Radar')
    
    # 4. Spectrum Reconstruction Examples (if data available)
    if 'data_samples' in results and 'real_spectra' in results['data_samples']:
        data_samples = results['data_samples']
        n_samples = min(3, len(data_samples['real_spectra']))
        frequencies = np.linspace(0.5, 3.0, data_samples['real_spectra'].shape[1])
        
        for i in range(n_samples):
            offset = i * 3  # Offset for visual separation
            axes[1, 1].plot(frequencies, data_samples['real_spectra'][i] + offset, 
                           label=f'Real {i+1}', linestyle='-', alpha=0.8)
            axes[1, 1].plot(frequencies, data_samples['pred_spectra'][i] + offset, 
                           label=f'Pred {i+1}', linestyle='--', alpha=0.8)
        
        axes[1, 1].set_xlabel('Frequency (THz)')
        axes[1, 1].set_ylabel('Transmission Coefficient (dB)')
        axes[1, 1].set_title('Spectrum Reconstruction Examples')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        axes[1, 1].text(0.5, 0.5, 'Spectrum reconstruction\nexamples not available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, alpha=0.7)
        axes[1, 1].set_title('Spectrum Reconstruction Examples')
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f"forward_model_evaluation_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Forward model evaluation plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate forward model performance")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    evaluate_forward_model(
        model_dir=args.model_dir,
        data_path=args.data_path,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
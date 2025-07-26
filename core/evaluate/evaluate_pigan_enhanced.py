# PI_GAN_THZ/core/evaluate/evaluate_pigan_enhanced.py

import sys
import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入统一评估器
from core.evaluate.unified_evaluator import UnifiedEvaluator
import config.config as cfg
from core.utils.set_seed import set_seed

# 导入增强模型
from core.models.enhanced_generator import EnhancedGenerator
from core.models.enhanced_discriminator import EnhancedDiscriminator
from core.models.enhanced_forward_model import EnhancedForwardPINN

def evaluate_pigan_enhanced(model_dir: str = None, 
                           data_path: str = None, 
                           num_samples: int = 1000):
    """
    评估增强版PI-GAN模型性能
    
    Args:
        model_dir: 模型目录路径
        data_path: 数据集路径
        num_samples: 评估样本数
    """
    print("\n--- 增强版PI-GAN Model Evaluation ---")
    
    # 设置随机种子
    set_seed(cfg.RANDOM_SEED)
    
    # 创建评估器
    evaluator = UnifiedEvaluator()
    
    # 加载模型和数据
    if not evaluator.load_models_enhanced(model_dir):  # 使用增强版模型加载方法
        print("❌ Failed to load enhanced models!")
        return
    
    if not evaluator.load_dataset(data_path):
        print("❌ Failed to load dataset!")
        return
    
    # 运行PI-GAN评估
    results = evaluator.evaluate_pigan_enhanced(num_samples)
    
    # 打印详细结果
    print("\n📊 增强版PI-GAN Evaluation Results:")
    print("-" * 50)
    
    # 生成器性能
    param_metrics = results['parameter_prediction']
    print("Generator - Parameter Prediction:")
    print(f"  - R²: {param_metrics['r2']:.6f}")
    print(f"  - MAE: {param_metrics['mae']:.6f}")
    print(f"  - RMSE: {param_metrics['rmse']:.6f}")
    print(f"  - Pearson R: {param_metrics['pearson_r']:.6f}")
    print(f"  - MAPE: {param_metrics['mape']:.2f}%")
    
    # 判别器性能
    disc_metrics = results['discriminator_performance']
    print("\nDiscriminator Performance:")
    print(f"  - Real Accuracy: {disc_metrics['real_accuracy']:.6f}")
    print(f"  - Fake Accuracy: {disc_metrics['fake_accuracy']:.6f}")
    print(f"  - Overall Accuracy: {disc_metrics['overall_accuracy']:.6f}")
    print(f"  - Real Score Mean: {disc_metrics['real_score_mean']:.6f}")
    print(f"  - Fake Score Mean: {disc_metrics['fake_score_mean']:.6f}")
    
    # 物理一致性性能
    physics_metrics = results['physics_consistency']
    print("\nPhysics Consistency:")
    print(f"  - Physics Score Mean: {physics_metrics['physics_score_mean']:.6f}")
    print(f"  - Physics Score Std: {physics_metrics['physics_score_std']:.6f}")
    
    # 性能评估
    print("\n🎯 Performance Assessment:")
    param_r2 = param_metrics['r2']
    disc_acc = disc_metrics['overall_accuracy']
    physics_score = physics_metrics['physics_score_mean']
    
    if param_r2 > 0.8 and disc_acc > 0.8 and physics_score < 0.3:
        print("✅ 增强版PI-GAN shows EXCELLENT performance!")
        print("  - Generator accurately predicts structural parameters")
        print("  - Discriminator effectively distinguishes real vs fake")
        print("  - Generated samples are physically consistent")
    elif param_r2 > 0.6 and disc_acc > 0.7 and physics_score < 0.5:
        print("✅ 增强版PI-GAN shows GOOD performance!")
        print("  - Generator performs well with room for improvement")
        print("  - Discriminator shows decent discrimination capability")
        print("  - Generated samples show acceptable physical consistency")
    elif param_r2 > 0.4 and disc_acc > 0.6 and physics_score < 0.7:
        print("⚠️ 增强版PI-GAN shows MODERATE performance.")
        print("  - Generator needs improvement in parameter prediction")
        print("  - Discriminator shows acceptable performance")
        print("  - Physical consistency needs improvement")
    else:
        print("❌ 增强版PI-GAN shows POOR performance and needs improvement.")
        print("  - Generator fails to accurately predict parameters")
        print("  - Discriminator shows poor discrimination capability")
        print("  - Generated samples lack physical consistency")
    
    # Generate evaluation plots
    plot_pigan_evaluation_enhanced(results)
    
    print(f"\n✅ 增强版PI-GAN evaluation completed with {results['num_samples']} samples.")

def plot_pigan_evaluation_enhanced(results):
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
    plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f"pigan_enhanced_evaluation_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Enhanced PI-GAN evaluation plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate enhanced PI-GAN model performance")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    evaluate_pigan_enhanced(
        model_dir=args.model_dir,
        data_path=args.data_path,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
# PI_GAN_THZ/core/evaluate/evaluate_structural_prediction.py

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

def evaluate_structural_prediction(model_dir: str = None, 
                                   data_path: str = None, 
                                   num_samples: int = 500):
    """
    评估结构预测能力
    
    Args:
        model_dir: 模型目录路径
        data_path: 数据集路径
        num_samples: 评估样本数
    """
    print("\n--- Structural Prediction Evaluation ---")
    
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
    
    # 运行结构预测评估
    results = evaluator.evaluate_structural_prediction(num_samples)
    
    # 打印详细结果
    print("\n📊 Structural Prediction Evaluation Results:")
    print("-" * 50)
    
    print("Parameter Constraint Validation:")
    print(f"  - Parameter Violation Rate: {results['param_range_violation_rate']:.6f}")
    print(f"  - Avg Violations per Sample: {results['avg_param_violations']:.6f}")
    
    print("\nReconstruction Quality:")
    print(f"  - Reconstruction Error Mean: {results['reconstruction_error_mean']:.6f}")
    print(f"  - Reconstruction Error Std: {results['reconstruction_error_std']:.6f}")
    
    print("\nPrediction Consistency:")
    print(f"  - Consistency Score Mean: {results['consistency_score_mean']:.6f}")
    print(f"  - Consistency Score Std: {results['consistency_score_std']:.6f}")
    
    # 性能评估
    print("\n🎯 Performance Assessment:")
    violation_rate = results['param_range_violation_rate']
    consistency = results['consistency_score_mean']
    recon_error = results['reconstruction_error_mean']
    
    if violation_rate < 0.05 and consistency > 0.9 and recon_error < 0.01:
        print("✅ Structural prediction shows EXCELLENT reliability!")
        print("  - Very low parameter constraint violations")
        print("  - High prediction consistency")
        print("  - Excellent spectrum reconstruction")
    elif violation_rate < 0.1 and consistency > 0.8 and recon_error < 0.05:
        print("✅ Structural prediction shows GOOD reliability!")
        print("  - Low parameter constraint violations")
        print("  - Good prediction consistency")
        print("  - Good spectrum reconstruction")
    elif violation_rate < 0.2 and consistency > 0.6 and recon_error < 0.1:
        print("⚠️ Structural prediction shows MODERATE reliability.")
        print("  - Acceptable parameter constraint violations")
        print("  - Moderate prediction consistency")
        print("  - Moderate spectrum reconstruction quality")
    else:
        print("❌ Structural prediction shows POOR reliability and needs improvement.")
        print("  - High parameter constraint violations")
        print("  - Low prediction consistency")
        print("  - Poor spectrum reconstruction quality")
    
    # 建议
    print("\n💡 Recommendations:")
    if violation_rate > 0.1:
        print("  - Consider adding stronger parameter constraints during training")
        print("  - Implement parameter clipping or regularization")
    
    if consistency < 0.7:
        print("  - Improve model stability through better training procedures")
        print("  - Consider ensemble methods for more consistent predictions")
    
    if recon_error > 0.05:
        print("  - Enhance forward model training for better reconstruction")
        print("  - Check data quality and preprocessing procedures")
    
    # Generate evaluation plots
    plot_structural_prediction_evaluation(results)
    
    print(f"\n✅ Structural prediction evaluation completed with {results['num_samples']} samples.")

def plot_structural_prediction_evaluation(results):
    """Generate plots for structural prediction evaluation"""
    
    # Set up English plotting
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Structural Prediction Evaluation Results', fontsize=16)
    
    # 1. Violation Rate Assessment
    violation_rate = results['param_range_violation_rate']
    target_rate = 0.1  # Target violation rate
    
    categories = ['Current\nViolation Rate', 'Target\nViolation Rate']
    values = [violation_rate, target_rate]
    colors = ['red' if violation_rate > target_rate else 'green', 'green']
    
    bars = axes[0, 0].bar(categories, values, color=colors, alpha=0.7)
    axes[0, 0].set_title('Parameter Violation Rate Assessment')
    axes[0, 0].set_ylabel('Violation Rate')
    axes[0, 0].set_ylim(0, max(violation_rate, target_rate) * 1.2)
    
    # Add value labels
    for bar, value in zip(bars, values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + height * 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Consistency and Reconstruction Quality
    metrics = ['Consistency\nScore', 'Reconstruction\nError (×10)']
    metric_values = [results['consistency_score_mean'], results['reconstruction_error_mean'] * 10]
    targets = [0.9, 0.5]  # Target values
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0, 1].bar(x - width/2, metric_values, width, label='Current', alpha=0.7, color='blue')
    bars2 = axes[0, 1].bar(x + width/2, targets, width, label='Target', alpha=0.7, color='green')
    
    axes[0, 1].set_title('Quality Metrics Comparison')
    axes[0, 1].set_ylabel('Metric Value')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics)
    axes[0, 1].legend()
    
    # Add value labels
    for bar, value in zip(bars1, metric_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Performance Summary Radar Chart
    ax_radar = plt.subplot(2, 2, 3, projection='polar')
    
    categories_radar = ['Low Violation\nRate', 'High Consistency', 'Low Recon\nError', 'Overall\nQuality']
    values_radar = [
        1 - min(violation_rate, 1.0),  # Inverted violation rate
        results['consistency_score_mean'],
        1 - min(results['reconstruction_error_mean'] * 10, 1.0),  # Inverted and scaled error
        (1 - min(violation_rate, 1.0) + results['consistency_score_mean'] + 
         1 - min(results['reconstruction_error_mean'] * 10, 1.0)) / 3  # Average quality
    ]
    
    angles = np.linspace(0, 2*np.pi, len(categories_radar), endpoint=False).tolist()
    values_radar += values_radar[:1]
    angles += angles[:1]
    
    ax_radar.plot(angles, values_radar, 'o-', linewidth=2, label='Current Performance', color='orange')
    ax_radar.fill(angles, values_radar, alpha=0.25, color='orange')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories_radar)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Structural Prediction Quality Radar')
    
    # 4. Detailed Statistics
    axes[1, 1].axis('off')
    
    # Create a text summary
    summary_text = f"""
Detailed Statistics:

Parameter Violations:
• Violation Rate: {violation_rate:.1%}
• Avg Violations/Sample: {results['avg_param_violations']:.2f}

Reconstruction Quality:
• Mean Error: {results['reconstruction_error_mean']:.6f}
• Error Std Dev: {results['reconstruction_error_std']:.6f}

Prediction Consistency:
• Mean Score: {results['consistency_score_mean']:.4f}
• Score Std Dev: {results['consistency_score_std']:.4f}

Performance Rating:
"""
    
    # Add performance rating
    if violation_rate < 0.05 and results['consistency_score_mean'] > 0.9:
        rating = "EXCELLENT ✅"
        color = 'green'
    elif violation_rate < 0.1 and results['consistency_score_mean'] > 0.8:
        rating = "GOOD ✅"
        color = 'blue'
    elif violation_rate < 0.2 and results['consistency_score_mean'] > 0.6:
        rating = "MODERATE ⚠️"
        color = 'orange'
    else:
        rating = "POOR ❌"
        color = 'red'
    
    summary_text += f"• Overall: {rating}"
    
    axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # Add colored rating box
    axes[1, 1].text(0.7, 0.15, rating, transform=axes[1, 1].transAxes,
                   fontsize=14, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
    
    plt.tight_layout()
    
    # Save plot
    plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(plots_dir, f"structural_prediction_evaluation_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Structural prediction evaluation plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate structural prediction capabilities")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    evaluate_structural_prediction(
        model_dir=args.model_dir,
        data_path=args.data_path,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
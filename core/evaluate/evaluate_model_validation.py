# PI_GAN_THZ/core/evaluate/evaluate_model_validation.py

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

def evaluate_model_validation(model_dir: str = None, 
                              data_path: str = None, 
                              num_samples: int = 500):
    """
    评估模型验证性能
    
    Args:
        model_dir: 模型目录路径
        data_path: 数据集路径
        num_samples: 评估样本数
    """
    print("\n--- Model Validation Evaluation ---")
    
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
    
    # 运行模型验证评估
    results = evaluator.evaluate_model_validation(num_samples)
    
    # 打印详细结果
    print("\n📊 Model Validation Results:")
    print("-" * 50)
    
    print("Cycle Consistency Analysis:")
    print(f"  - Cycle Consistency Error Mean: {results['cycle_consistency_error_mean']:.8f}")
    print(f"  - Cycle Consistency Error Std: {results['cycle_consistency_error_std']:.8f}")
    
    print("\nPrediction Stability Analysis:")
    print(f"  - Prediction Stability Mean: {results['prediction_stability_mean']:.8f}")
    print(f"  - Prediction Stability Std: {results['prediction_stability_std']:.8f}")
    
    print("\nPhysical Plausibility Analysis:")
    print(f"  - Physical Plausibility Mean: {results['physical_plausibility_mean']:.6f}")
    print(f"  - Physical Plausibility Std: {results['physical_plausibility_std']:.6f}")
    
    # 性能评估
    print("\n🎯 Performance Assessment:")
    cycle_error = results['cycle_consistency_error_mean']
    stability = results['prediction_stability_mean']
    plausibility = results['physical_plausibility_mean']
    
    print("\nCycle Consistency:")
    if cycle_error < 0.001:
        print("✅ EXCELLENT cycle consistency - Model maintains perfect round-trip accuracy")
    elif cycle_error < 0.01:
        print("✅ GOOD cycle consistency - Model shows reliable round-trip performance")
    elif cycle_error < 0.05:
        print("⚠️ MODERATE cycle consistency - Some information loss in round-trip")
    else:
        print("❌ POOR cycle consistency - Significant information loss detected")
    
    print("\nPrediction Stability:")
    if stability < 0.001:
        print("✅ EXCELLENT stability - Model is highly robust to input noise")
    elif stability < 0.01:
        print("✅ GOOD stability - Model shows good noise tolerance")
    elif stability < 0.05:
        print("⚠️ MODERATE stability - Model somewhat sensitive to input variations")
    else:
        print("❌ POOR stability - Model highly sensitive to input noise")
    
    print("\nPhysical Plausibility:")
    if plausibility > 0.9:
        print("✅ EXCELLENT physical plausibility - Predictions are highly realistic")
    elif plausibility > 0.8:
        print("✅ GOOD physical plausibility - Predictions are generally realistic")
    elif plausibility > 0.6:
        print("⚠️ MODERATE physical plausibility - Some unrealistic predictions")
    else:
        print("❌ POOR physical plausibility - Many unrealistic predictions")
    
    # 整体评估
    print("\n🎯 Overall Model Validation:")
    excellent_count = sum([
        cycle_error < 0.01,
        stability < 0.01,
        plausibility > 0.8
    ])
    
    if excellent_count == 3:
        print("🌟 EXCELLENT - Model passes all validation tests with high scores!")
        print("  - Ready for production use")
        print("  - High confidence in predictions")
    elif excellent_count >= 2:
        print("✅ GOOD - Model passes most validation tests!")
        print("  - Suitable for most applications")
        print("  - Monitor performance in practice")
    elif excellent_count >= 1:
        print("⚠️ MODERATE - Model shows mixed validation results.")
        print("  - Use with caution in critical applications")
        print("  - Consider additional training or constraints")
    else:
        print("❌ POOR - Model fails multiple validation tests.")
        print("  - Not recommended for production use")
        print("  - Requires significant improvements")
    
    # 建议
    print("\n💡 Recommendations:")
    if cycle_error > 0.01:
        print("  - Improve training with cycle consistency loss")
        print("  - Check forward model accuracy")
        print("  - Verify data preprocessing pipeline")
    
    if stability > 0.01:
        print("  - Add noise regularization during training")
        print("  - Implement dropout or other regularization techniques")
        print("  - Consider model ensemble for stability")
    
    if plausibility < 0.8:
        print("  - Add physical constraint losses")
        print("  - Implement parameter range validation")
        print("  - Review training data quality")
    
    # Generate evaluation plots
    plot_model_validation_evaluation(results)
    
    print(f"\n✅ Model validation completed with {results['num_samples']} samples.")

def plot_model_validation_evaluation(results):
    """Generate plots for model validation evaluation"""
    
    # Set up English plotting
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Validation Evaluation Results', fontsize=16)
    
    # 1. Validation Metrics Overview
    metrics = ['Cycle\nConsistency\nError', 'Prediction\nStability', 'Physical\nPlausibility']
    current_values = [
        results['cycle_consistency_error_mean'],
        results['prediction_stability_mean'],
        results['physical_plausibility_mean']
    ]
    targets = [0.01, 0.01, 0.8]  # Target values
    
    # Normalize for visualization (lower is better for first two, higher for last)
    normalized_current = [
        1 - min(current_values[0] / 0.1, 1.0),  # Cycle consistency (inverted)
        1 - min(current_values[1] / 0.1, 1.0),  # Stability (inverted)
        current_values[2]  # Plausibility (higher is better)
    ]
    normalized_targets = [0.9, 0.9, 0.8]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, normalized_current, width, label='Current', alpha=0.7, color='blue')
    bars2 = axes[0, 0].bar(x + width/2, normalized_targets, width, label='Target', alpha=0.7, color='green')
    
    axes[0, 0].set_title('Validation Metrics (Normalized)')
    axes[0, 0].set_ylabel('Quality Score (0-1)')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metrics)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars1, normalized_current):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Error Metrics (Raw Values)
    error_metrics = ['Cycle Error', 'Stability Error']
    error_values = [results['cycle_consistency_error_mean'], results['prediction_stability_mean']]
    error_targets = [0.01, 0.01]
    
    colors = ['red' if val > target else 'green' for val, target in zip(error_values, error_targets)]
    
    bars = axes[0, 1].bar(error_metrics, error_values, color=colors, alpha=0.7)
    axes[0, 1].axhline(y=0.01, color='green', linestyle='--', alpha=0.7, label='Target (0.01)')
    axes[0, 1].set_title('Error Metrics (Raw Values)')
    axes[0, 1].set_ylabel('Error Value')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    
    # Add value labels
    for bar, value in zip(bars, error_values):
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height * 1.1,
                       f'{value:.6f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Validation Quality Radar Chart
    ax_radar = plt.subplot(2, 2, 3, projection='polar')
    
    categories_radar = ['Low Cycle\nError', 'High Stability', 'High\nPlausibility', 'Overall\nQuality']
    values_radar = [
        1 - min(results['cycle_consistency_error_mean'] / 0.1, 1.0),
        1 - min(results['prediction_stability_mean'] / 0.1, 1.0),
        results['physical_plausibility_mean'],
        (normalized_current[0] + normalized_current[1] + normalized_current[2]) / 3
    ]
    
    angles = np.linspace(0, 2*np.pi, len(categories_radar), endpoint=False).tolist()
    values_radar += values_radar[:1]
    angles += angles[:1]
    
    ax_radar.plot(angles, values_radar, 'o-', linewidth=2, label='Current Performance', color='purple')
    ax_radar.fill(angles, values_radar, alpha=0.25, color='purple')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(categories_radar)
    ax_radar.set_ylim(0, 1)
    ax_radar.set_title('Model Validation Quality Radar')
    
    # 4. Detailed Statistics and Assessment
    axes[1, 1].axis('off')
    
    # Create detailed summary
    summary_text = f"""
Model Validation Statistics:

Cycle Consistency:
• Error Mean: {results['cycle_consistency_error_mean']:.6f}
• Error Std: {results['cycle_consistency_error_std']:.6f}

Prediction Stability:
• Stability Mean: {results['prediction_stability_mean']:.6f}
• Stability Std: {results['prediction_stability_std']:.6f}

Physical Plausibility:
• Plausibility Mean: {results['physical_plausibility_mean']:.4f}
• Plausibility Std: {results['physical_plausibility_std']:.4f}

Overall Assessment:
"""
    
    # Performance assessment
    cycle_error = results['cycle_consistency_error_mean']
    stability = results['prediction_stability_mean']
    plausibility = results['physical_plausibility_mean']
    
    if cycle_error < 0.01 and stability < 0.01 and plausibility > 0.8:
        rating = "EXCELLENT ✅"
        color = 'green'
    elif cycle_error < 0.05 and stability < 0.05 and plausibility > 0.6:
        rating = "GOOD ✅"
        color = 'blue'
    elif cycle_error < 0.1 and stability < 0.1 and plausibility > 0.4:
        rating = "MODERATE ⚠️"
        color = 'orange'
    else:
        rating = "POOR ❌"
        color = 'red'
    
    summary_text += f"• Model Validation: {rating}"
    
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
    plot_path = os.path.join(plots_dir, f"model_validation_evaluation_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Model validation evaluation plot saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model validation performance")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    evaluate_model_validation(
        model_dir=args.model_dir,
        data_path=args.data_path,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
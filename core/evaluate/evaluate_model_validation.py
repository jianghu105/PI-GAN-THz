# PI_GAN_THZ/core/evaluate/evaluate_model_validation.py

import sys
import os
import torch
import numpy as np
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入统一评估器
from unified_evaluator import UnifiedEvaluator
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
    
    print(f"\n✅ Model validation completed with {results['num_samples']} samples.")

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
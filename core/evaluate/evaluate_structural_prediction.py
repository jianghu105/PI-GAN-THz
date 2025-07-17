# PI_GAN_THZ/core/evaluate/evaluate_structural_prediction.py

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
    
    print(f"\n✅ Structural prediction evaluation completed with {results['num_samples']} samples.")

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
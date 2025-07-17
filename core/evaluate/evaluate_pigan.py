# PI_GAN_THZ/core/evaluate/evaluate_pigan.py

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

def evaluate_pigan(model_dir: str = None, 
                   data_path: str = None, 
                   num_samples: int = 1000):
    """
    评估PI-GAN模型性能
    
    Args:
        model_dir: 模型目录路径
        data_path: 数据集路径
        num_samples: 评估样本数
    """
    print("\n--- PI-GAN Model Evaluation ---")
    
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
    
    # 运行PI-GAN评估
    results = evaluator.evaluate_pigan(num_samples)
    
    # 打印详细结果
    print("\n📊 PI-GAN Evaluation Results:")
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
    
    # 性能评估
    print("\n🎯 Performance Assessment:")
    param_r2 = param_metrics['r2']
    disc_acc = disc_metrics['overall_accuracy']
    
    if param_r2 > 0.8 and disc_acc > 0.8:
        print("✅ PI-GAN shows EXCELLENT performance!")
        print("  - Generator accurately predicts structural parameters")
        print("  - Discriminator effectively distinguishes real vs fake")
    elif param_r2 > 0.6 and disc_acc > 0.7:
        print("✅ PI-GAN shows GOOD performance!")
        print("  - Generator performs well with room for improvement")
        print("  - Discriminator shows decent discrimination capability")
    elif param_r2 > 0.4 and disc_acc > 0.6:
        print("⚠️ PI-GAN shows MODERATE performance.")
        print("  - Generator needs improvement in parameter prediction")
        print("  - Discriminator shows acceptable performance")
    else:
        print("❌ PI-GAN shows POOR performance and needs improvement.")
        print("  - Generator fails to accurately predict parameters")
        print("  - Discriminator shows poor discrimination capability")
    
    print(f"\n✅ PI-GAN evaluation completed with {results['num_samples']} samples.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PI-GAN model performance")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    evaluate_pigan(
        model_dir=args.model_dir,
        data_path=args.data_path,
        num_samples=args.num_samples
    )

if __name__ == "__main__":
    main()
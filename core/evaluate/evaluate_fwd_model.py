# PI_GAN_THZ/core/evaluate/evaluate_fwd_model.py

import sys
import os
import torch
import numpy as np
import argparse

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入统一评估器
from unified_evaluator import UnifiedEvaluator
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
    
    print(f"\n✅ Forward model evaluation completed with {results['num_samples']} samples.")

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
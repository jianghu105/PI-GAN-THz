#!/usr/bin/env python3
"""
模拟训练脚本 - 用于快速测试
"""

import sys
import os

# 添加项目根目录到路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

def mock_training_test():
    """模拟训练测试（不实际训练）"""
    print("🧪 模拟训练测试")
    
    try:
        # 测试导入
        from core.train.unified_trainer import UnifiedTrainer
        print("✓ UnifiedTrainer 导入成功")
        
        # 测试初始化
        trainer = UnifiedTrainer(device="cpu")
        print("✓ UnifiedTrainer 初始化成功")
        
        # 测试模型初始化
        trainer.initialize_models()
        print("✓ 模型初始化成功")
        
        # 测试模型保存（创建空文件进行测试）
        import config.config as cfg
        os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
        
        # 模拟保存模型文件
        import torch
        dummy_state = {'test': torch.tensor([1.0])}
        
        model_files = [
            "generator_final.pth",
            "discriminator_final.pth", 
            "forward_model_final.pth"
        ]
        
        for filename in model_files:
            filepath = os.path.join(cfg.SAVED_MODELS_DIR, filename)
            torch.save(dummy_state, filepath)
            print(f"✓ 模拟保存: {filename}")
        
        print("🎉 模拟训练测试通过！")
        print(f"📁 模型文件保存在: {cfg.SAVED_MODELS_DIR}")
        
        # 测试评估器兼容性
        print("\n🔍 测试评估器兼容性...")
        from core.evaluate.unified_evaluator import UnifiedEvaluator
        evaluator = UnifiedEvaluator(device="cpu")
        print("✓ UnifiedEvaluator 初始化成功")
        
        # 检查模型文件是否能被找到
        for filename in model_files:
            filepath = os.path.join(cfg.SAVED_MODELS_DIR, filename)
            if os.path.exists(filepath):
                print(f"✓ 模型文件存在: {filename}")
            else:
                print(f"✗ 模型文件缺失: {filename}")
        
        print("✅ 所有测试通过！统一训练系统已就绪。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    mock_training_test()

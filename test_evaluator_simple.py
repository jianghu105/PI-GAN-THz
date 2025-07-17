#!/usr/bin/env python3
# 简化的评估器测试

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_config():
    """测试配置文件"""
    print("="*60)
    print("测试配置文件")
    print("="*60)
    
    try:
        import config.config as cfg
        print(f"✅ 配置文件导入成功")
        print(f"项目根目录: {cfg.PROJECT_ROOT}")
        print(f"数据集路径: {cfg.DATASET_PATH}")
        print(f"数据文件存在: {os.path.exists(cfg.DATASET_PATH)}")
        return True
    except Exception as e:
        print(f"❌ 配置文件导入失败: {e}")
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n" + "="*60)
    print("测试数据加载器")
    print("="*60)
    
    try:
        import config.config as cfg
        from core.utils.data_loader import MetamaterialDataset
        
        # 测试数据集类初始化（不加载数据）
        dataset = MetamaterialDataset(
            data_path=cfg.DATASET_PATH, 
            num_points_per_sample=cfg.SPECTRUM_DIM,
            load_data=False  # 不加载数据，只初始化
        )
        print(f"✅ 数据集类初始化成功")
        
        # 测试实际数据加载
        dataset_with_data = MetamaterialDataset(
            data_path=cfg.DATASET_PATH, 
            num_points_per_sample=cfg.SPECTRUM_DIM,
            load_data=True  # 加载数据
        )
        print(f"✅ 数据加载成功，样本数: {len(dataset_with_data)}")
        return True
        
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_models():
    """测试模型导入"""
    print("\n" + "="*60)
    print("测试模型导入")
    print("="*60)
    
    try:
        from core.models.generator import Generator
        from core.models.discriminator import Discriminator  
        from core.models.forward_model import ForwardModel
        print("✅ 模型类导入成功")
        return True
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        return False

def main():
    print("PI-GAN 评估器简化测试")
    print("=" * 60)
    
    success = True
    
    # 测试配置
    if not test_config():
        success = False
    
    # 测试数据加载器
    if not test_data_loader():
        success = False
    
    # 测试模型导入
    if not test_models():
        success = False
    
    print("\n" + "="*60)
    if success:
        print("🎉 所有测试通过！")
    else:
        print("💥 部分测试失败！")
    
    return success

if __name__ == "__main__":
    main()
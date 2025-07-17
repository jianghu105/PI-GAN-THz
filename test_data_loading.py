#!/usr/bin/env python3
# 测试数据加载功能

import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

import config.config as cfg
from core.utils.data_loader import MetamaterialDataset

def test_data_loading():
    print("="*60)
    print("测试数据加载功能")
    print("="*60)
    
    # 打印配置信息
    print(f"项目根目录: {cfg.PROJECT_ROOT}")
    print(f"数据目录: {cfg.DATA_DIR}")
    print(f"数据集路径: {cfg.DATASET_PATH}")
    print(f"文件是否存在: {os.path.exists(cfg.DATASET_PATH)}")
    
    if not os.path.exists(cfg.DATASET_PATH):
        print("❌ 数据文件不存在！")
        return False
    
    try:
        # 尝试加载数据集
        print("\n正在加载数据集...")
        dataset = MetamaterialDataset(
            data_path=cfg.DATASET_PATH, 
            num_points_per_sample=cfg.SPECTRUM_DIM
        )
        
        print(f"✅ 数据集加载成功！")
        print(f"样本数量: {len(dataset)}")
        print(f"光谱维度: {len(dataset.spectrum_cols)}")
        print(f"参数列: {dataset.param_cols}")
        print(f"指标列: {dataset.metric_cols}")
        
        # 测试获取一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\n样本结构:")
            print(f"  - 光谱形状: {sample[0].shape}")
            print(f"  - 参数形状: {sample[1].shape}")
            print(f"  - 归一化参数形状: {sample[2].shape}")
            print(f"  - 指标形状: {sample[3].shape}")
            print(f"  - 归一化指标形状: {sample[4].shape}")
            
        return True
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False

if __name__ == "__main__":
    success = test_data_loading()
    if success:
        print("\n🎉 数据加载测试通过！")
    else:
        print("\n💥 数据加载测试失败！")
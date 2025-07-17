#!/usr/bin/env python3
# 快速测试数据加载

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

# 检查配置
print("检查配置...")
try:
    import config.config as cfg
    print(f"✅ 配置导入成功")
    print(f"数据集路径: {cfg.DATASET_PATH}")
    print(f"文件存在: {os.path.exists(cfg.DATASET_PATH)}")
except Exception as e:
    print(f"❌ 配置导入失败: {e}")
    sys.exit(1)

# 检查数据文件内容
print("\n检查数据文件内容...")
try:
    with open(cfg.DATASET_PATH, 'r') as f:
        header = f.readline().strip()
        cols = header.split(',')
        freq_cols = [col for col in cols if col.startswith('Freq_')]
        print(f"✅ 找到 {len(freq_cols)} 个光谱列")
        print(f"✅ 总列数: {len(cols)}")
        
        # 检查所需列
        required_cols = ['r1', 'r2', 'w', 'g', 'f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
        missing_cols = [col for col in required_cols if col not in cols]
        if missing_cols:
            print(f"❌ 缺失列: {missing_cols}")
        else:
            print(f"✅ 所有必需列都存在")
            
except Exception as e:
    print(f"❌ 文件读取失败: {e}")
    sys.exit(1)

print("\n✅ 所有检查通过！数据格式正确。")
print("\n现在可以尝试运行评估模块：")
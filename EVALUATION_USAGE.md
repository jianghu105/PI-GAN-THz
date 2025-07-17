# PI-GAN 评估系统使用指南

## 🎯 系统概述

PI-GAN 评估系统现已完全重构，包含4个核心评估模块和完整的可视化系统：

1. **前向网络评估** - 光谱和指标预测性能
2. **PI-GAN评估** - 生成器和判别器性能
3. **结构预测评估** - 参数约束和一致性
4. **模型验证** - 循环一致性和物理合理性

## 📊 可视化功能

所有评估结果都会自动生成可视化图表，包括：
- 性能指标对比图
- 雷达图和散点图
- 误差分布直方图
- 综合评估摘要
- 详细评估报告

所有图表和报告保存到 `plots/` 目录。

## 🚀 使用方法

### 1. 运行完整评估（推荐）

```bash
# 运行所有4个评估模块 + 自动生成可视化
python core/evaluate/unified_evaluator.py --num_samples 1000

# 指定模型目录和数据路径
python core/evaluate/unified_evaluator.py \
    --model_dir saved_models \
    --data_path dataset/THz_Metamaterial_Spectra_With_Metrics.csv \
    --num_samples 1000
```

**输出文件：**
- `plots/forward_network_evaluation_YYYYMMDD_HHMMSS.png`
- `plots/pigan_evaluation_YYYYMMDD_HHMMSS.png`
- `plots/structural_prediction_evaluation_YYYYMMDD_HHMMSS.png`
- `plots/model_validation_evaluation_YYYYMMDD_HHMMSS.png`
- `plots/comprehensive_summary_YYYYMMDD_HHMMSS.png`
- `plots/unified_evaluation_report.txt`

### 2. 运行单独评估模块

#### 前向网络评估
```bash
python core/evaluate/evaluate_fwd_model.py
```

#### PI-GAN评估
```bash
python core/evaluate/evaluate_pigan.py
```

#### 结构预测评估
```bash
python core/evaluate/evaluate_structural_prediction.py
```

#### 模型验证评估
```bash
python core/evaluate/evaluate_model_validation.py
```

## 📈 评估指标

### 前向网络评估
- **光谱预测**：R²、MAE、RMSE、Pearson相关系数
- **指标预测**：R²、MAE、RMSE、Pearson相关系数
- **目标**：R² > 0.85

### PI-GAN评估
- **参数预测**：R²、MAE、RMSE、Pearson相关系数
- **判别器性能**：真实样本准确率、生成样本准确率
- **目标**：参数R² > 0.80，判别器准确率 > 0.80

### 结构预测评估
- **参数违约率**：超出物理范围的参数比例
- **重建误差**：光谱重建质量
- **一致性得分**：预测一致性
- **目标**：违约率 < 10%，一致性 > 0.95

### 模型验证
- **循环一致性**：spectrum → params → spectrum的误差
- **预测稳定性**：噪声扰动下的预测稳定性
- **物理合理性**：参数的物理约束满足程度
- **目标**：循环误差 < 0.01，稳定性 < 0.001，合理性 > 0.80

## 🎨 可视化内容

### 1. 前向网络可视化
- 性能指标对比（R²、MAE）
- 详细指标雷达图
- 光谱重建对比
- 误差分布直方图
- 性能等级评估

### 2. PI-GAN可视化
- 生成器和判别器性能
- 参数预测散点图
- 得分分布直方图
- 对抗训练平衡分析

### 3. 结构预测可视化
- 参数违约分析
- 约束满足度评估
- 重建误差分布
- 一致性得分分析

### 4. 模型验证可视化
- 循环一致性分析
- 稳定性测试结果
- 物理合理性评估
- 综合验证摘要

### 5. 综合摘要可视化
- 所有模块性能对比
- 目标达成情况
- 改进建议
- 整体评级

## 🔧 配置选项

### 命令行参数
```bash
python core/evaluate/unified_evaluator.py \
    --model_dir saved_models \           # 模型目录
    --data_path dataset/data.csv \       # 数据路径
    --num_samples 1000 \                 # 评估样本数
    --device cuda \                      # 计算设备
    --seed 42                            # 随机种子
```

### 评估配置
可以修改 `config/config.py` 中的评估参数：
- `EVALUATION_BATCH_SIZE`：评估批次大小
- `EVALUATION_NUM_SAMPLES`：默认评估样本数
- `EVALUATION_METRICS`：评估指标设置

## 📋 输出解读

### 控制台输出示例
```
=== Forward Network Evaluation (1000 samples) ===
✓ Forward network evaluation completed
  - Spectrum R²: 0.5018
  - Metrics R²: 0.8037

=== PI-GAN Evaluation (1000 samples) ===
✓ PI-GAN evaluation completed
  - Parameter R²: 0.5329
  - Discriminator Accuracy: 0.6085

🎨 Generating evaluation visualizations...
✓ Forward network evaluation plot saved: plots/forward_network_evaluation_20250718_070634.png
✓ PI-GAN evaluation plot saved: plots/pigan_evaluation_20250718_070634.png
✓ Structural prediction evaluation plot saved: plots/structural_prediction_evaluation_20250718_070634.png
✓ Model validation evaluation plot saved: plots/model_validation_evaluation_20250718_070634.png
✓ Comprehensive summary plot saved: plots/comprehensive_summary_20250718_070634.png
🎯 All evaluation visualizations generated in: plots

Evaluation report saved to: plots/unified_evaluation_report.txt
```

### 性能等级
- **优秀** (绿色)：达到或超过目标值
- **良好** (蓝色)：接近目标值
- **中等** (黄色)：有待改进
- **较差** (红色)：需要重点优化

## 🔍 问题诊断

### 常见问题
1. **模型文件未找到**
   - 检查 `saved_models/` 目录
   - 确保模型文件存在：`generator_final.pth`、`discriminator_final.pth`、`forward_model_final.pth`

2. **数据加载失败**
   - 检查数据路径：`dataset/THz_Metamaterial_Spectra_With_Metrics.csv`
   - 确保数据格式正确

3. **GPU内存不足**
   - 降低批次大小：`--batch_size 32`
   - 减少样本数：`--num_samples 500`

4. **可视化生成失败**
   - 检查 `plots/` 目录权限
   - 确保matplotlib依赖安装

## 🎯 优化建议

根据评估结果进行模型优化：

### 当前性能问题（基于上次评估）
1. **前向网络光谱预测R² = 0.50**（目标 > 0.85）
2. **PI-GAN参数预测R² = 0.53**（目标 > 0.80）
3. **参数违约率 = 87.4%**（目标 < 10%）
4. **物理合理性 = 0.13**（目标 > 0.80）

### 推荐优化方案
```bash
# 1. 使用优化训练器重新训练
python core/train/optimized_trainer.py --epochs 200

# 2. 查看优化配置
python config/training_optimization.py

# 3. 重新评估
python core/evaluate/unified_evaluator.py --num_samples 1000
```

详细优化指南参见：`optimization_guide.md`

## 📞 技术支持

如需帮助，请检查：
1. `README_EVALUATION.md` - 评估系统详细说明
2. `optimization_guide.md` - 模型优化指南
3. `logs/` 目录 - 训练和评估日志
4. `plots/` 目录 - 可视化结果
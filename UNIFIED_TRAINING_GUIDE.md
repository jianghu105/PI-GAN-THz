# PI-GAN 统一训练系统指南

## 🎯 问题解决

**上次训练错误分析：**
- ❌ 模型保存文件名不匹配：`generator_optimized.pth` vs `generator_final.pth`
- ✅ 已修复：现在保存为评估器期望的文件名

## 🚀 统一训练器特性

### 新的统一训练器 (`unified_trainer.py`) 整合了：

1. **前向模型预训练** (`pretrain_fwd_model.py`)
2. **PI-GAN训练** (`train_pigan.py`) 
3. **优化训练** (`optimized_trainer.py`)

### 三种训练模式：

#### 1️⃣ 仅前向模型训练
```bash
python core/train/unified_trainer.py --mode forward_only --forward_epochs 100
```

#### 2️⃣ 仅PI-GAN训练
```bash
python core/train/unified_trainer.py --mode pigan_only --pigan_epochs 200
```

#### 3️⃣ 完整训练流水线（推荐）
```bash
python core/train/unified_trainer.py --mode full --forward_epochs 50 --pigan_epochs 200
```

## 📊 训练流程详解

### 完整训练流水线 (`--mode full`)

```
阶段1: 前向模型预训练 (50轮)
├── 目标：建立准确的参数→光谱映射
├── 损失：光谱重建 + 指标预测 + 平滑性
└── 输出：训练好的前向模型

阶段2: PI-GAN对抗训练 (200轮) 
├── 判别器训练：区分真实/生成参数对
├── 生成器训练：多重损失优化
│   ├── 对抗损失 (权重1.0)
│   ├── 重建损失 (权重10.0) ⭐ 最重要
│   ├── 约束损失 (权重3.0)
│   ├── 物理损失 (权重2.0)
│   └── 稳定性损失 (权重1.0)
└── 输出：完整训练的PI-GAN系统
```

## 🎛️ 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | `full` | 训练模式：`forward_only`, `pigan_only`, `full` |
| `--forward_epochs` | `50` | 前向模型训练轮数 |
| `--pigan_epochs` | `200` | PI-GAN训练轮数 |
| `--batch_size` | `64` | 批次大小 |
| `--device` | `auto` | 设备：`auto`, `cpu`, `cuda` |
| `--seed` | `42` | 随机种子 |
| `--resume` | `None` | 从检查点恢复 |

## 📈 输出文件

### 训练完成后会生成：

**模型文件** (saved_models/):
- ✅ `generator_final.pth` (评估器可直接加载)
- ✅ `discriminator_final.pth`
- ✅ `forward_model_final.pth`
- 📄 `*_unified.pth` (备份文件)

**训练曲线** (plots/):
- 📊 `unified_training_curves_full_YYYYMMDD_HHMMSS.png`
- 包含：损失曲线、违约率、学习率等

**检查点** (checkpoints/):
- 💾 每50轮自动保存训练状态

## 🔍 实时监控

### 训练过程中会显示：

```
=== Forward Model Training (50 epochs) ===
Epoch [10/50] - Loss: 0.012345, LR: 0.000100
Epoch [20/50] - Loss: 0.008234, LR: 0.000095
...

=== PI-GAN Training (200 epochs) ===
Epoch [10/200]
  G Loss: 0.456789 | D Loss: 0.234567
  Violation Rate: 0.8740
  G LR: 0.000200
...
```

### 关键指标监控：
- **G Loss**: 生成器损失 (应稳定下降)
- **D Loss**: 判别器损失 (应与G Loss平衡)
- **Violation Rate**: 约束违约率 (目标：87.4% → <10%)
- **Learning Rate**: 学习率调度

## 💡 使用建议

### 快速开始（推荐）：
```bash
# 完整训练流水线
python core/train/unified_trainer.py --mode full

# 训练完成后立即评估
python core/evaluate/unified_evaluator.py --num_samples 1000
```

### 高级用法：

**调整训练参数：**
```bash
python core/train/unified_trainer.py \
    --mode full \
    --forward_epochs 100 \
    --pigan_epochs 300 \
    --batch_size 32 \
    --seed 123
```

**从检查点恢复：**
```bash
python core/train/unified_trainer.py \
    --mode full \
    --resume checkpoints/unified_checkpoint_full_epoch_100.pth
```

**分阶段训练：**
```bash
# 第一步：只训练前向模型
python core/train/unified_trainer.py --mode forward_only --forward_epochs 100

# 第二步：基于预训练前向模型训练PI-GAN
python core/train/unified_trainer.py --mode pigan_only --pigan_epochs 200
```

## 🎯 预期改进

使用统一训练器后，预期性能提升：

| 指标 | 当前值 | 目标值 | 改进策略 |
|------|--------|--------|----------|
| 参数违约率 | 87.4% | <10% | 强化约束损失 |
| 生成器R² | 0.53 | >0.80 | 多重损失优化 |
| 前向网络R² | 0.50 | >0.85 | 预训练+平滑损失 |
| 物理合理性 | 0.13 | >0.80 | 物理约束损失 |

## 🐛 故障排除

### 常见问题：

**1. 模型文件未找到**
- ✅ 已修复：统一训练器保存正确的文件名
- 检查：`saved_models/` 目录下应有 `*_final.pth` 文件

**2. GPU内存不足**
```bash
# 解决方案：减少批次大小
python core/train/unified_trainer.py --mode full --batch_size 32
```

**3. 训练不稳定**
- 检查违约率是否过高
- 查看训练曲线是否震荡
- 考虑降低学习率

**4. 评估器加载失败**
- 确保训练完成后有 `*_final.pth` 文件
- 检查文件路径配置

## 📞 完整工作流程

### 从训练到评估的完整流程：

```bash
# 1. 完整训练
python core/train/unified_trainer.py --mode full

# 2. 完整评估
python core/evaluate/unified_evaluator.py --num_samples 1000

# 3. 检查结果
ls plots/  # 查看生成的图表
cat plots/unified_evaluation_report.txt  # 查看评估报告
```

### 预期输出目录结构：
```
PI-GAN-THz/
├── saved_models/
│   ├── generator_final.pth      ✅ 评估器可加载
│   ├── discriminator_final.pth  ✅ 评估器可加载
│   └── forward_model_final.pth  ✅ 评估器可加载
├── plots/
│   ├── unified_training_curves_full_*.png
│   ├── forward_network_evaluation_*.png
│   ├── pigan_evaluation_*.png
│   └── unified_evaluation_report.txt
└── checkpoints/
    └── unified_checkpoint_*.pth
```

现在训练和评估系统完全整合，解决了模型文件名不匹配的问题！
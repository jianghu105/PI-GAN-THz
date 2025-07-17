# PI-GAN 评估模块使用指南

## 🔧 问题修复总结

### 修复的问题：
1. **配置文件数据路径错误**：`DATASET_PATH` 从 `"THZ.txt"` 修正为 `"THz_Metamaterial_Spectra_With_Metrics.csv"`
2. **导入路径错误**：修正了各评估模块中统一评估器的导入路径
3. **依赖兼容性**：配置文件现在可以在没有PyTorch的环境下也能工作

### 修复的文件：
- `config/config.py` - 数据路径和依赖兼容性
- `core/evaluate/evaluate_fwd_model.py` - 导入路径
- `core/evaluate/evaluate_pigan.py` - 导入路径  
- `core/evaluate/evaluate_structural_prediction.py` - 导入路径
- `core/evaluate/evaluate_model_validation.py` - 导入路径

## 📁 评估模块结构

```
core/evaluate/
├── unified_evaluator.py              # 核心统一评估器
├── evaluate_fwd_model.py            # 前向网络评估
├── evaluate_pigan.py                # PI-GAN评估
├── evaluate_structural_prediction.py # 结构预测评估
└── evaluate_model_validation.py     # 模型验证评估
```

## 🚀 使用方法

### 1. 完整评估（推荐）
```bash
python core/evaluate/unified_evaluator.py --num_samples 1000
```

### 2. 单独模块评估

#### 前向网络评估
```bash
python core/evaluate/evaluate_fwd_model.py --num_samples 1000
```

#### PI-GAN模型评估  
```bash
python core/evaluate/evaluate_pigan.py --num_samples 1000
```

#### 结构预测评估
```bash
python core/evaluate/evaluate_structural_prediction.py --num_samples 500
```

#### 模型验证评估
```bash
python core/evaluate/evaluate_model_validation.py --num_samples 500
```

## 📊 评估内容

### 1. 前向网络评估
- **光谱预测准确度**：输入参数→预测THz光谱
- **指标预测准确度**：输入参数→预测性能指标
- **评估指标**：R²、MAE、RMSE、Pearson相关系数、MAPE

### 2. PI-GAN评估
- **生成器性能**：光谱→结构参数预测准确度
- **判别器性能**：真实vs生成样本识别准确率
- **对抗性能分析**：得分分布和判别能力

### 3. 结构预测评估
- **参数约束验证**：预测参数合理性检查
- **重建质量评估**：光谱重建误差分析
- **预测一致性**：模型稳定性评估

### 4. 模型验证评估
- **循环一致性**：光谱→参数→光谱往返准确性
- **预测稳定性**：噪声扰动下的鲁棒性
- **物理合理性**：预测结果的物理可行性

## 📈 评估输出

每个模块会输出：
- **实时控制台信息**：评估进度和中间结果
- **详细性能指标**：具体数值评估结果  
- **性能等级评估**：优秀/良好/中等/较差分级
- **改进建议**：针对性优化建议
- **评估报告文件**：保存到 `saved_models/` 目录

## 🔍 故障排除

### 常见问题：

1. **数据加载失败**
   - 检查 `dataset/THz_Metamaterial_Spectra_With_Metrics.csv` 文件是否存在
   - 验证数据格式是否正确

2. **模型加载失败**  
   - 确保 `saved_models/` 目录包含训练好的模型文件：
     - `generator_final.pth`
     - `discriminator_final.pth` 
     - `forward_model_final.pth`

3. **导入错误**
   - 确保从项目根目录运行脚本
   - 检查Python路径设置

### 测试命令：
```bash
# 快速配置测试
python quick_test.py

# 简化评估器测试  
python test_evaluator_simple.py
```

## 📋 命令行参数

所有评估模块支持以下参数：
- `--model_dir`: 模型目录路径（默认：`saved_models/`）
- `--data_path`: 数据集路径（默认：配置文件路径）
- `--num_samples`: 评估样本数（默认：各模块不同）
- `--device`: 计算设备（默认：auto）
- `--seed`: 随机种子（默认：42）

## ✅ 验证评估系统

运行以下命令验证系统工作正常：

```bash
# 1. 测试配置和数据
python quick_test.py

# 2. 测试简化评估器
python test_evaluator_simple.py

# 3. 运行单个评估模块测试
python core/evaluate/evaluate_fwd_model.py --num_samples 10
```

现在评估系统已经修复并可以正常使用了！🎉
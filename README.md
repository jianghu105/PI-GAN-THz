# PI-GAN-THz: 基于物理信息GAN的太赫兹超材料逆向优化设计

## 项目概述

本项目旨在开发一个深度学习模型，实现对太赫兹（THz）超材料的逆向优化设计。我们采用了一种结合物理信息神经网络（PINN）和生成对抗网络（GAN）的创新方法，以高效、准确地生成具有特定光谱响应的超材料结构参数。

**核心组件包括：**
*   **前向物理代理模型 (Forward Model)**：一个神经网络，用于模拟超材料结构参数到其透射光谱的映射，实现结构-频谱的物理验证。
*   **生成器 (Generator)**：负责生成超材料的结构参数。
*   **判别器 (Discriminator)**：区分真实的（来自数据集）和伪造的（由生成器生成并经前向模型验证）结构-光谱-特征对，评估生成数据的逼真度。
*   **物理信息损失 (Physics-Informed Loss)**：间接促使生成器生成符合物理规律的结构，确保生成设计的物理有效性。
*   **光谱特征提取器 (Spectral Feature Extractor)**：从透射光谱中提取关键共振特征（如谐振频率、Q因子、FoM、灵敏度等）。

## 文件结构

```
PI_GAN_THz/
├─ core/
│   ├─ models/         # 神经网络模型定义（生成器、判别器、前向模型）
│   │   ├─ forward_model.py    # 前向物理代理模型及光谱特征提取器
│   │   ├─ generator.py        # 生成器模型
│   │   └─ discriminator.py    # 判别器模型
│   ├─ train/          # 训练脚本
│   │   ├─ pretrain_forward_model.py # 预训练前向模型
│   │   └─ train_pigan.py            # PI-GAN对抗训练脚本
│   ├─ evaluate/       # 评估与测试脚本
│   │   └─ evaluator.py              # 模型评估脚本
│   └─ utils/          # 工具函数
│       ├─ data_loader.py      # 数据加载、预处理与归一化
│       └─ loss.py             # 损失函数定义（包括物理信息损失和梯度惩罚）
├─ config/             # 配置文件
│   └─ config.py               # 全局配置参数
├─ dataset/            # 数据集及说明
│   └─ THz_Metamaterial_Spectra_With_Metrics.csv # 原始数据集
│   └─ thz_data_processor.py # 原始数据处理脚本（特征提取逻辑参考）
├─ checkpoints/        # 训练中间权重保存目录
├─ saved_models/       # 最优模型和最终模型保存目录
├─ generated_data/     # 生成数据和评估图表输出目录
├─ requirements.txt    # Python依赖包列表
├─ README.md           # 项目说明
```

## 环境依赖与安装

建议使用Python 3.8+。主要依赖包已列于 `requirements.txt`。

```bash
pip install -r requirements.txt
```

## 数据准备

数据集 `dataset/THz_Metamaterial_Spectra_With_Metrics.csv` 包含了4450条超材料数据，每条数据包括：
*   **结构参数**: `r1`, `r2`, `w`, `g`
*   **关键共振特征**: `f1`, `f2`, `Q1`, `FoM1`, `S1`, `Q2`, `FoM2`, `S2`
*   **透射光谱数据**: `Freq_0.50`, `Freq_0.51`, ..., `Freq_2.99` (250个数据点)

数据加载、归一化和划分由 `core/utils/data_loader.py` 自动处理。

## 模型架构

### 前向物理代理模型 (`core/models/forward_model.py`)
*   **网络结构**: 接收结构参数作为输入，通过多层全连接网络预测250维透射光谱。最后一层使用 `Sigmoid` 激活函数，假设光谱已归一化到 [0, 1] 范围。
*   **光谱特征提取器 (`SpectralFeatureExtractor`)**: 一个不可训练的模块，用于从预测光谱中提取 `f1`, `f2`, `Q1`, `FoM1`, `S1`, `Q2`, `FoM2`, `S2` 等物理指标。该模块已优化，能够稳健地处理批处理数据并避免 `NaN` 值。

### 生成器 (`core/models/generator.py`)
*   **网络结构**: 接收一个随机噪声向量（latent vector）作为输入，通过多层全连接网络生成超材料的结构参数。网络中包含 `BatchNorm1d` 和 `LeakyReLU` 激活函数，最后一层使用 `Sigmoid` 确保输出在 [0, 1] 范围内（与归一化后的结构参数匹配）。

### 判别器 (`core/models/discriminator.py`)
*   **网络结构**: 接收拼接后的结构参数、透射光谱和物理特征作为输入，通过多层全连接网络判断输入数据的真实性。网络中包含 `BatchNorm1d` 和 `LeakyReLU` 激活函数，最后一层没有激活函数（适用于WGAN-GP）。

## 快速开始

### 1. 预训练前向模型

在训练PI-GAN之前，需要预训练前向模型，使其能够准确地预测光谱。

```bash
python3 core/train/pretrain_forward_model.py
```
预训练完成后，最佳模型将保存到 `saved_models/best_forward_model.pth`。

### 2. 训练PI-GAN

运行以下脚本开始PI-GAN的对抗训练。训练过程中会加载预训练好的前向模型并冻结其参数。

```bash
python3 core/train/train_pigan.py
```
训练过程中，模型权重将定期保存到 `saved_models/` 目录。

### 3. 评估模型

训练完成后，可以使用评估脚本来分析生成模型的性能。该脚本将生成新的样本，计算其统计摘要，并绘制与真实数据对比的分布图和光谱对比图。

```bash
python3 core/evaluate/evaluator.py
```
评估结果（统计摘要和图表）将保存到 `generated_data/` 目录。

## 关键概念

*   **物理信息神经网络 (PINN)**: 将物理定律或领域知识编码到神经网络的损失函数中，以指导模型学习符合物理规律的解。在本项中，前向模型作为物理代理，其预测结果用于构建物理损失。
*   **生成对抗网络 (GAN)**: 由一个生成器和一个判别器组成，两者通过对抗性训练相互学习。生成器试图生成逼真的数据，判别器则试图区分真实数据和生成数据。
*   **Wasserstein GAN with Gradient Penalty (WGAN-GP)**: 一种改进的GAN架构，通过使用Wasserstein距离和梯度惩罚来提高训练的稳定性和生成样本的质量，有效缓解了传统GAN训练中的模式崩塌和梯度消失问题。
*   **物理信息损失 (Physics-Informed Loss)**: 在本项目的GAN框架中，生成器的损失函数不仅包含对抗性损失，还包含基于前向模型预测的物理损失（生成结构预测光谱与目标光谱的MSE）和指标损失（生成结构预测指标与目标指标的MSE）。这确保了生成器不仅能生成“看起来真实”的结构，还能生成“物理上合理”的结构。

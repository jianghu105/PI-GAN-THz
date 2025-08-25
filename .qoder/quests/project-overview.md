# PI_GAN_THz 项目概览

## 项目简介

PI_GAN_THz是一个基于物理信息生成对抗网络（Physics-Informed GAN）的太赫兹超材料逆向设计框架。该项目旨在解决传统电磁仿真优化方法效率低、耗时长的问题，通过深度学习实现从目标光谱特征反向生成符合物理规律的超材料结构参数。

### 核心价值
- **高效逆向设计**: 从目标光谱特征生成可行的超材料结构参数
- **物理约束保证**: 通过物理信息损失确保生成结构的物理有效性
- **端到端流程**: 提供完整的训练、评估与可视化工作流

### 应用场景
- 超材料结构优化设计
- 太赫兹器件性能预测
- 电磁特性快速仿真
- 材料科学研究辅助

## 技术架构

### 整体架构设计

```mermaid
graph TB
    subgraph "数据层"
        A[THzDataset<br/>4450条样本]
        B[数据预处理<br/>归一化]
    end
    
    subgraph "模型层"
        C[前向模型<br/>ForwardModel]
        D[生成器<br/>Generator]
        E[判别器<br/>Discriminator]
    end
    
    subgraph "训练层"
        F[前向模型预训练<br/>pretrain_forward_model.py]
        G[PI-GAN对抗训练<br/>train_pigan.py]
    end
    
    subgraph "评估层"
        H[模型评估<br/>evaluator.py]
        I[前向模型评估<br/>evaluate_forward_model.py]
    end
    
    A --> B
    B --> F
    B --> G
    F --> C
    C --> G
    D --> G
    E --> G
    G --> H
    F --> I
```

### 两阶段训练流程

#### 第一阶段：前向模型预训练
```mermaid
graph LR
    A[结构参数] --> B[MLP编码器<br/>1024维潜在向量]
    B --> C[CNN解码器<br/>256维光谱]
    C --> D[光谱裁剪<br/>250维]
    D --> E[物理特征提取器<br/>谐振频率、Q因子等]
    E --> F[MSE损失<br/>光谱+指标]
```

#### 第二阶段：PI-GAN对抗训练
```mermaid
graph TD
    subgraph "生成器分支"
        A[噪声向量z] --> B[生成器G]
        C[条件向量c] --> B
        B --> D[生成结构参数]
    end
    
    subgraph "前向模型分支"
        D --> E[前向模型F<br/>冻结参数]
        E --> F[预测光谱]
        E --> G[预测指标]
    end
    
    subgraph "判别器分支"
        D --> H[判别器D]
        F --> H
        G --> H
        I[真实数据] --> H
        H --> J[真实性评分]
        H --> K[物理误差反馈]
    end
    
    subgraph "损失计算"
        J --> L[WGAN-GP损失]
        K --> M[物理信息损失]
        L --> N[优化生成器]
        M --> N
    end
```

### 核心组件详解

#### 前向模型（ForwardModel）
- **功能**: 结构参数 → 光谱响应映射
- **架构**: MLP编码器 + CNN解码器
- **输入**: 超材料结构参数
- **输出**: 透射光谱 + 物理指标（谐振频率、Q因子、FoM、灵敏度）

```mermaid
classDiagram
    class ForwardModel {
        +input_dim: int
        +output_dim: int
        +spectra_network: Sequential
        +decoder_cnn: Sequential
        +feature_extractor: DifferentiableSpectralFeatureExtractor
        +calibrator: MetricsCalibrator
        +forward(struct_params): (spectra, metrics)
    }
    
    class DifferentiableSpectralFeatureExtractor {
        +freq_grid: Tensor
        +log_alpha: Parameter
        +log_gamma: Parameter
        +log_sigma_frac: Parameter
        +forward(spectra_batch): Tensor
    }
    
    ForwardModel --> DifferentiableSpectralFeatureExtractor
```

#### 生成器（Generator）
- **功能**: 条件生成结构参数
- **输入**: 随机噪声 + 条件向量（目标物理指标）
- **输出**: 超材料结构参数
- **架构**: 深层MLP网络

#### 判别器（Discriminator）
- **功能**: 多模态真实性判别 + 物理误差反馈
- **架构**: 三分支网络设计

```mermaid
classDiagram
    class Discriminator {
        +struct_encoder: Sequential
        +spectra_encoder: Sequential
        +metric_encoder: Sequential
        +fusion: Sequential
        +real_fake_head: Sequential
        +physical_error_head: Sequential
        +forward(combined_input): (real_score, physical_error)
    }
    
    class PhysicalConstraintModule {
        +min_bounds: Tensor
        +max_bounds: Tensor
        +forward(struct_params): penalty
    }
    
    Discriminator --> PhysicalConstraintModule
```

### 物理信息损失机制

#### 损失函数组成
```mermaid
graph TD
    A[物理信息损失] --> B[对抗损失<br/>WGAN-GP]
    A --> C[物理预测损失<br/>光谱MSE]
    A --> D[指标匹配损失<br/>物理指标MSE]
    A --> E[物理误差反馈损失<br/>判别器反馈]
    
    B --> F[确保生成真实性]
    C --> G[保证光谱准确性]
    D --> H[匹配目标指标]
    E --> I[物理约束合理性]
```

#### 损失权重配置
| 损失类型 | 权重 | 作用 |
|---------|------|------|
| 对抗损失 | 1.0 | 基础真实性 |
| 物理预测损失 | 10.0 | 光谱匹配 |
| 指标匹配损失 | 5.0 | 性能指标 |
| 物理误差反馈 | 2.0 | 约束违反惩罚 |

## 数据架构

### 数据集特征
- **样本数量**: 4450条
- **输入维度**: 结构参数（多维）
- **输出维度**: 250维光谱 + 4维物理指标

### 数据预处理流程
```mermaid
graph LR
    A[原始数据] --> B[数据清洗]
    B --> C[特征归一化<br/>MinMaxScaler]
    C --> D[数据分割<br/>训练/验证/测试]
    D --> E[批处理加载<br/>DataLoader]
```

### 物理指标提取
```mermaid
graph TD
    A[光谱数据] --> B[峰值检测]
    B --> C[谐振频率识别]
    C --> D[Q因子计算]
    D --> E[FoM计算]
    E --> F[灵敏度分析]
    F --> G[物理指标向量]
```

## 配置管理

### 全局配置架构
```mermaid
graph TB
    A[config.py] --> B[模型参数]
    A --> C[训练参数]
    A --> D[数据路径]
    A --> E[设备配置]
    
    B --> F[前向模型配置]
    B --> G[生成器配置]
    B --> H[判别器配置]
    
    C --> I[学习率设置]
    C --> J[批大小配置]
    C --> K[训练轮数]
    
    D --> L[数据集路径]
    D --> M[模型保存路径]
    D --> N[日志输出路径]
```

### 关键参数配置

#### 模型结构参数
| 参数名 | 默认值 | 说明 |
|-------|--------|------|
| LATENT_DIM | 128 | 生成器潜在空间维度 |
| CONDITION_DIM | 4 | 条件向量维度 |
| STRUCT_DIM | variable | 结构参数维度 |
| SPECTRA_DIM | 250 | 光谱维度 |

#### 训练超参数
| 参数名 | 默认值 | 说明 |
|-------|--------|------|
| LEARNING_RATE | 0.0002 | 学习率 |
| BATCH_SIZE | 64 | 批大小 |
| NUM_EPOCHS | 1000 | 训练轮数 |
| LAMBDA_GP | 10.0 | 梯度惩罚权重 |

## 训练与评估策略

### 训练策略

#### 前向模型预训练
```mermaid
graph LR
    A[加载数据] --> B[模型初始化]
    B --> C[MSE损失训练]
    C --> D[验证集评估]
    D --> E{性能达标?}
    E -->|否| C
    E -->|是| F[保存最佳模型]
```

#### PI-GAN对抗训练
```mermaid
graph LR
    A[加载预训练模型] --> B[冻结前向模型]
    B --> C[生成器训练]
    C --> D[判别器训练]
    D --> E[物理约束检查]
    E --> F{收敛判断}
    F -->|否| C
    F -->|是| G[保存最终模型]
```

### 评估指标

#### 前向模型评估
| 指标 | 计算方式 | 目标值 |
|------|----------|--------|
| 光谱MSE | MSE(pred_spectra, true_spectra) | < 0.01 |
| 指标MAE | MAE(pred_metrics, true_metrics) | < 5% |
| R²得分 | R²(pred, true) | > 0.95 |

#### 生成器评估
| 指标 | 计算方式 | 目标值 |
|------|----------|--------|
| 物理有效性 | 约束违反率 | < 1% |
| 光谱相似度 | 余弦相似度 | > 0.9 |
| 指标匹配度 | 相对误差 | < 10% |

## 模块测试策略

### 单元测试覆盖
```mermaid
graph TB
    A[测试框架] --> B[模型测试]
    A --> C[数据测试]
    A --> D[训练测试]
    
    B --> E[前向模型测试]
    B --> F[生成器测试]
    B --> G[判别器测试]
    
    C --> H[数据加载测试]
    C --> I[预处理测试]
    C --> J[特征提取测试]
    
    D --> K[训练流程测试]
    D --> L[损失计算测试]
    D --> M[模型保存测试]
```

### 测试用例设计
- **模型前向传播测试**: 验证输入输出维度一致性
- **损失函数测试**: 确保梯度计算正确性
- **数据管道测试**: 验证数据预处理流程
- **物理约束测试**: 检查生成结构的物理合理性
- **端到端测试**: 完整训练流程验证

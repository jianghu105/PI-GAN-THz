# PI_GAN_THZ/config/training_optimization.py
# 训练优化配置 - 基于评估结果的针对性改进

import os
import torch

# 从原配置继承基本设置
from config.config import *

# =============================================================================
# 优化配置 - 基于评估结果的改进方案
# =============================================================================

# --- 1. 前向网络优化 (解决光谱预测R²=0.5018问题) ---
FORWARD_MODEL_OPTIMIZATION = {
    # 增强网络容量
    'hidden_dims': [128, 256, 512, 1024, 512, 256],  # 更深更宽的网络
    'dropout_rate': 0.3,                              # 防止过拟合
    'batch_norm': True,                               # 批归一化稳定训练
    'activation': 'leaky_relu',                       # 更好的激活函数
    
    # 优化损失函数
    'spectrum_loss_weight': 1.0,                      # 光谱重建损失权重
    'metrics_loss_weight': 0.8,                       # 指标预测损失权重
    'smoothness_loss_weight': 0.1,                    # 光谱平滑性损失
    'physics_loss_weight': 0.2,                       # 物理约束损失
    
    # 训练参数
    'learning_rate': 1e-4,                            # 降低学习率
    'epochs': 200,                                    # 增加训练轮数
    'early_stopping_patience': 20,                   # 早停耐心
    'lr_scheduler': 'cosine',                         # 余弦学习率调度
}

# --- 2. 生成器优化 (解决参数预测R²=0.5329问题) ---
GENERATOR_OPTIMIZATION = {
    # 网络结构优化
    'hidden_dims': [512, 1024, 2048, 1024, 512, 256], # 更强的表征能力
    'residual_blocks': 3,                             # 残差连接
    'attention_layers': 2,                            # 注意力机制
    'dropout_rate': 0.2,                              # 适度正则化
    
    # 损失函数改进
    'adversarial_loss_weight': 1.0,                   # 对抗损失
    'reconstruction_loss_weight': 10.0,               # 重建损失（提高权重）
    'perceptual_loss_weight': 5.0,                    # 感知损失
    'constraint_loss_weight': 2.0,                    # 参数约束损失
    
    # 训练策略
    'learning_rate': 2e-4,                            # 适中学习率
    'beta1': 0.5,                                     # Adam优化器参数
    'beta2': 0.999,
    'gradient_clip': 1.0,                             # 梯度裁剪
}

# --- 3. 判别器优化 (解决判别准确率=0.6085问题) ---
DISCRIMINATOR_OPTIMIZATION = {
    # 网络结构
    'hidden_dims': [256, 512, 1024, 512, 256, 128],  # 平衡的网络深度
    'spectral_norm': True,                            # 谱归一化稳定训练
    'dropout_rate': 0.3,                              # 防止过拟合
    'leaky_relu_slope': 0.2,                          # LeakyReLU斜率
    
    # 训练参数
    'learning_rate': 1e-4,                            # 略低于生成器
    'label_smoothing': 0.1,                           # 标签平滑
    'instance_noise': 0.05,                           # 实例噪声
    
    # 损失函数
    'loss_type': 'wgan_gp',                           # WGAN-GP损失
    'gradient_penalty_weight': 10.0,                  # 梯度惩罚权重
}

# --- 4. 参数约束优化 (解决违约率87.4%问题) ---
CONSTRAINT_OPTIMIZATION = {
    # 硬约束
    'parameter_clipping': True,                       # 参数裁剪
    'parameter_ranges': {                             # 严格参数范围
        'r1': (2.2, 2.8),
        'r2': (2.2, 2.8), 
        'w': (2.2, 2.8),
        'g': (2.2, 2.8)
    },
    
    # 软约束损失
    'range_penalty_weight': 5.0,                      # 范围惩罚权重
    'boundary_smoothness': 0.1,                       # 边界平滑度
    'constraint_activation': 'sigmoid',               # 约束激活函数
    
    # 物理约束
    'physics_constraint_weight': 3.0,                 # 物理约束权重
    'resonance_constraint': True,                     # 谐振约束
    'causality_constraint': True,                     # 因果约束
}

# --- 5. 训练过程优化 ---
TRAINING_OPTIMIZATION = {
    # 数据增强
    'data_augmentation': {
        'noise_level': 0.05,                          # 噪声增强
        'frequency_shift': 0.02,                      # 频率偏移
        'amplitude_scale': 0.1,                       # 幅度缩放
    },
    
    # 训练策略
    'progressive_training': True,                     # 渐进训练
    'curriculum_learning': True,                      # 课程学习
    'mixed_precision': True,                          # 混合精度
    
    # 评估策略
    'evaluation_frequency': 10,                      # 评估频率
    'save_best_model': True,                          # 保存最优模型
    'validation_split': 0.2,                         # 验证集比例
    
    # 超参数调度
    'warmup_epochs': 10,                              # 预热轮数
    'cosine_annealing': True,                         # 余弦退火
    'weight_decay': 1e-4,                             # 权重衰减
}

# --- 6. 损失函数权重优化 ---
LOSS_WEIGHTS = {
    # 基础损失
    'adversarial_loss': 1.0,                          # 对抗损失
    'reconstruction_loss': 10.0,                      # 重建损失
    'forward_consistency_loss': 5.0,                  # 前向一致性损失
    
    # 约束损失
    'parameter_constraint_loss': 3.0,                 # 参数约束损失
    'physics_constraint_loss': 2.0,                   # 物理约束损失
    'smoothness_loss': 1.0,                           # 平滑性损失
    
    # 正则化损失
    'diversity_loss': 0.5,                            # 多样性损失
    'sparsity_loss': 0.1,                             # 稀疏性损失
    'stability_loss': 1.0,                            # 稳定性损失
}

# --- 7. 模型架构优化 ---
MODEL_ARCHITECTURE = {
    'generator': {
        'base_channels': 64,
        'max_channels': 512,
        'num_residual_blocks': 6,
        'use_attention': True,
        'attention_heads': 8,
        'use_self_attention': True,
    },
    
    'discriminator': {
        'base_channels': 32,
        'max_channels': 256,
        'num_layers': 6,
        'use_spectral_norm': True,
        'use_gradient_penalty': True,
    },
    
    'forward_model': {
        'hidden_layers': [128, 256, 512, 1024, 512, 256, 128],
        'use_residual': True,
        'use_batch_norm': True,
        'use_dropout': True,
    }
}

# --- 8. 优化器配置 ---
OPTIMIZER_CONFIG = {
    'generator': {
        'type': 'adam',
        'lr': 2e-4,
        'betas': (0.5, 0.999),
        'weight_decay': 1e-4,
        'eps': 1e-8,
    },
    
    'discriminator': {
        'type': 'adam', 
        'lr': 1e-4,
        'betas': (0.5, 0.999),
        'weight_decay': 1e-4,
        'eps': 1e-8,
    },
    
    'forward_model': {
        'type': 'adam',
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'weight_decay': 1e-4,
        'eps': 1e-8,
    }
}

# --- 9. 评估目标 ---
EVALUATION_TARGETS = {
    'forward_network': {
        'spectrum_r2_target': 0.9,                    # 光谱预测R²目标
        'metrics_r2_target': 0.9,                     # 指标预测R²目标
    },
    
    'pigan': {
        'parameter_r2_target': 0.85,                  # 参数预测R²目标
        'discriminator_accuracy_target': 0.85,        # 判别器准确率目标
    },
    
    'structural_prediction': {
        'violation_rate_target': 0.05,                # 违约率目标<5%
        'consistency_score_target': 0.95,             # 一致性分数目标>95%
    },
    
    'model_validation': {
        'cycle_consistency_target': 0.005,            # 循环一致性目标<0.5%
        'stability_target': 0.001,                    # 稳定性目标<0.1%
        'plausibility_target': 0.9,                   # 物理合理性目标>90%
    }
}

# --- 10. 训练监控 ---
MONITORING_CONFIG = {
    'tensorboard_logging': True,                     # TensorBoard日志
    'wandb_logging': False,                          # Weights & Biases日志
    'checkpoint_frequency': 20,                      # 检查点频率
    'plot_frequency': 50,                            # 绘图频率
    'evaluation_frequency': 10,                      # 评估频率
    'early_stopping_patience': 30,                   # 早停耐心
    'save_best_only': True,                          # 只保存最优模型
}

# =============================================================================
# 导出优化配置
# =============================================================================

def get_optimization_config():
    """获取完整的优化配置"""
    return {
        'forward_model': FORWARD_MODEL_OPTIMIZATION,
        'generator': GENERATOR_OPTIMIZATION,
        'discriminator': DISCRIMINATOR_OPTIMIZATION,
        'constraints': CONSTRAINT_OPTIMIZATION,
        'training': TRAINING_OPTIMIZATION,
        'loss_weights': LOSS_WEIGHTS,
        'model_architecture': MODEL_ARCHITECTURE,
        'optimizer': OPTIMIZER_CONFIG,
        'evaluation_targets': EVALUATION_TARGETS,
        'monitoring': MONITORING_CONFIG,
    }

def print_optimization_summary():
    """打印优化配置摘要"""
    print("="*60)
    print("PI-GAN 优化配置摘要")
    print("="*60)
    print("🎯 优化目标:")
    print(f"  - 前向网络光谱预测R²: 0.50 → 0.90")
    print(f"  - PI-GAN参数预测R²: 0.53 → 0.85") 
    print(f"  - 判别器准确率: 0.61 → 0.85")
    print(f"  - 参数违约率: 87.4% → <5%")
    print(f"  - 物理合理性: 0.13 → 0.90")
    print("")
    print("🔧 主要改进:")
    print("  - 增强网络架构和容量")
    print("  - 优化损失函数权重")
    print("  - 加强参数约束机制")
    print("  - 改进训练策略")
    print("  - 添加物理约束损失")
    print("="*60)

if __name__ == "__main__":
    print_optimization_summary()
# PI_GAN_THZ/core/utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从新的物理约束模块导入损失函数
from .physics_constraints import (
    lc_frequency_constraint,
    maxwell_constraint,
    energy_conservation_constraint
)

# --- 1. GAN 核心损失 (保留，可能会在判别器训练中使用) ---

def criterion_bce():
    """
    二元交叉熵损失，用于判别器和生成器的对抗性训练。
    Returns:
        torch.nn.BCEWithLogitsLoss: BCE 损失函数实例，包含 Sigmoid 激活。
                                   通常用于直接输出 logit 的模型，避免数值不稳定。
                                   如果模型最后一层已使用 Sigmoid，则使用 nn.BCELoss。
    """
    # 我们的判别器最后已经用了 Sigmoid，所以这里用 BCELoss
    return nn.BCELoss()

def criterion_mse():
    """
    均方误差损失，用于重建任务和物理约束。
    Returns:
        torch.nn.MSELoss: MSE 损失函数实例。
    """
    return nn.MSELoss()

# --- 2. 新的生成器总损失函数 ---

def generator_total_loss(generator, discriminator, forward_model, z, target_spectrum, epoch, total_epochs=200):
    """
    生成器总损失函数 - 融合物理约束与对抗训练
    
    关键设计:
      1. 三阶段动态平衡: 早期侧重光谱匹配, 后期强化物理约束
      2. 针对您的SRR结构定制权重
      3. 金层0.2μm固定的物理模型简化
    
    返回:
      total_loss: 生成器总损失
      loss_components: 各损失分量字典
    """
    # 1. 生成参数
    # 注意：根据您的设计，生成器可能不需要 target_spectrum 作为输入
    # 这里假设它需要 z (随机噪声)
    params = generator(z)
    
    # 2. 通过前向网络获取指标
    # 假设 forward_model 返回 (spectrum, metrics)
    recon_spectrum, metrics = forward_model(params, return_metrics=True)
    f1_pred, q1_pred, _, s1_pred, f2_pred, q2_pred, _, s2_pred = metrics.unbind(1)
    
    # 3. 物理约束损失
    lc_loss = lc_frequency_constraint(params, f1_pred, f2_pred)
    maxwell_loss = maxwell_constraint(params, s1_pred, s2_pred)
    energy_loss = energy_conservation_constraint(q1_pred, q2_pred)
    
    # 4. 对抗损失
    # 判别器评估的是 (光谱, 参数) 对的真实性
    gan_loss = -torch.mean(discriminator(recon_spectrum, params))
    
    # 5. 光谱重建损失 (L1 损失对异常值更鲁棒)
    spectral_loss = F.l1_loss(recon_spectrum, target_spectrum)
    
    # 6. 动态权重 (针对您的SRR结构定制)
    lambda_data = max(0.8 - 0.006 * epoch, 0.2)  # 从0.8→0.2
    lambda_lc = min(3.0 * epoch / 50, 3.0)  # 从0→3.0
    lambda_maxwell = min(1.5 * max(epoch - 30, 0) / 70, 1.5)  # 30轮后启动
    lambda_energy = min(1.0 * max(epoch - 50, 0) / 50, 1.0)  # 50轮后启动
    
    # 7. 总损失
    total_loss = (
        gan_loss + 
        lambda_data * spectral_loss + 
        lambda_lc * lc_loss + 
        lambda_maxwell * maxwell_loss + 
        lambda_energy * energy_loss
    )
    
    loss_components = {
        'gan': gan_loss.item(),
        'spectral': spectral_loss.item(),
        'lc': lc_loss.item(),
        'maxwell': maxwell_loss.item(),
        'energy': energy_loss.item(),
        'lambda_data': lambda_data,
        'lambda_lc': lambda_lc,
        'lambda_maxwell': lambda_maxwell,
        'lambda_energy': lambda_energy
    }

    return total_loss, loss_components

# --- 3. BNN KL 损失 (保留，以备将来使用) ---

def bnn_kl_loss(model: nn.Module):
    """
    BNN (Bayesian Neural Network) KL 散度损失。
    对于本项目中采用的 nn.Dropout 结合 MC Dropout 的简化方法，此函数返回 0。
    """
    return torch.zeros(1, device=next(model.parameters()).device)

# PI_GAN_THZ/core/utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. GAN 核心损失 ---
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

# --- 2. 物理信息和重建损失 ---

def maxwell_equation_loss(predicted_spectrum: torch.Tensor, frequencies: torch.Tensor, predicted_params_norm: torch.Tensor):
    """
    麦克斯韦方程组约束损失。
    这是一个占位符函数，需要根据实际的物理模型进行详细实现。
    例如，可以鼓励光谱的平滑性、特定的边界条件，或与仿真器数据的梯度匹配等。
    
    此处为简化的示例：惩罚光谱剧烈波动 (鼓励平滑性)，并考虑参数对整体形状的影响。
    
    Args:
        predicted_spectrum (torch.Tensor): 预测的太赫兹光谱 (batch_size, num_spectrum_points)。
        frequencies (torch.Tensor): 对应的频率点 (num_spectrum_points)。
        predicted_params_norm (torch.Tensor): 归一化的预测结构参数 (batch_size, 4)。
    Returns:
        torch.Tensor: 麦克斯韦方程组损失。
    """
    # 示例1：光谱平滑性惩罚（二阶导数）
    # 使用 F.conv1d 或简单的差分计算离散二阶导数
    # 为了避免边界问题，可以对内部点进行计算
    if predicted_spectrum.size(1) < 3: # 至少需要3个点才能计算二阶导数
        return torch.zeros(1, device=predicted_spectrum.device) # 如果点太少，返回0损失
        
    # 计算一阶差分
    diff1 = predicted_spectrum[:, 1:] - predicted_spectrum[:, :-1]
    # 计算二阶差分 (近似二阶导数)
    diff2 = diff1[:, 1:] - diff1[:, :-1]
    
    # 对二阶导数进行 L2 惩罚，鼓励平滑性
    smoothness_loss = torch.mean(diff2**2)
    
    # 示例2：结合参数的某种约束 (占位符，需要具体物理意义)
    # 例如，可以要求光谱的某些特征（如平均透射率）与参数有特定关系
    # mean_transmission = torch.mean(predicted_spectrum, dim=1)
    # param_influence_loss = torch.mean((mean_transmission - (predicted_params_norm[:, 0] * 0.1 + predicted_params_norm[:, 1] * 0.05))**2)
    
    # 暂时只返回平滑性损失作为麦克斯韦方程组约束的简单代理
    return smoothness_loss # + param_influence_loss (如果添加了)


def lc_model_approx_loss(f1_pred_norm: torch.Tensor, f2_pred_norm: torch.Tensor, structural_params_norm: torch.Tensor):
    """
    LC 模型近似约束损失。
    惩罚预测的共振频率 (已归一化) 与基于结构参数的理论或近似物理关系之间的偏差。
    此处的“理论”关系是简化的线性模型，需要根据实际物理原理或数据分析进行构建和验证。
    
    Args:
        f1_pred_norm (torch.Tensor): 预测的第一个归一化共振频率 (batch_size, 1)。
        f2_pred_norm (torch.Tensor): 预测的第二个归一化共振频率 (batch_size, 1)。
        structural_params_norm (torch.Tensor): 归一化结构参数 (batch_size, 4)。
            结构参数顺序假定为：r1 (idx 0), r2 (idx 1), w (idx 2), g (idx 3)。
    Returns:
        torch.Tensor: LC 模型近似损失。
    """
    # 提取归一化参数
    r1_norm = structural_params_norm[:, 0].unsqueeze(1)
    r2_norm = structural_params_norm[:, 1].unsqueeze(1)
    w_norm = structural_params_norm[:, 2].unsqueeze(1)
    g_norm = structural_params_norm[:, 3].unsqueeze(1)

    # 简化的理论关系示例 (需要根据真实物理推导和数据分析)
    # 假设 f1 主要与 r1 和 w 有关，f2 主要与 r2 和 g 有关
    # 这些系数 (0.4, 0.6, 0.3, 0.7等) 需要通过对数据或物理模型的分析来确定。
    # 它们应该将归一化后的结构参数映射到归一化后的频率范围。
    theoretical_f1_norm = 0.4 * r1_norm + 0.6 * w_norm # + 其他参数的影响
    theoretical_f2_norm = 0.3 * r2_norm + 0.7 * g_norm # + 其他参数的影响

    # 为了使理论值更贴近实际，可以考虑添加偏置或更复杂的非线性项
    # 例如：theoretical_f1_norm = 0.4 * r1_norm + 0.6 * w_norm + 0.1 # 添加偏置

    # 计算 MSE 损失
    loss_f1 = F.mse_loss(f1_pred_norm, theoretical_f1_norm)
    loss_f2 = F.mse_loss(f2_pred_norm, theoretical_f2_norm)
    
    return loss_f1 + loss_f2


def structural_param_range_loss(predicted_params_norm: torch.Tensor):
    """
    结构参数范围约束损失。
    惩罚生成器预测的参数超出其归一化范围 [0, 1] (或原始物理范围)。
    
    Args:
        predicted_params_norm (torch.Tensor): 归一化的预测结构参数 (batch_size, 4)。
    Returns:
        torch.Tensor: 范围损失。
    """
    # 对小于0和大于1的部分进行惩罚
    # torch.clamp(input, min, max) 将输入张量的值限制在 [min, max] 范围内
    # 如果值在 [0, 1] 之间，则 (val - 0)^2 和 (val - 1)^2 都为 0 或很小
    # 如果 val < 0, 则 (val - 0)^2 > 0
    # 如果 val > 1, 则 (val - 1)^2 > 0
    
    # 惩罚超出下限的部分
    lower_bound_penalty = torch.clamp(0 - predicted_params_norm, min=0)**2
    # 惩罚超出上限的部分
    upper_bound_penalty = torch.clamp(predicted_params_norm - 1, min=0)**2
    
    # 对所有超出范围的惩罚取平均
    loss = torch.mean(lower_bound_penalty + upper_bound_penalty)
    return loss

def bnn_kl_loss(model: nn.Module):
    """
    BNN (Bayesian Neural Network) KL 散度损失。
    当使用 Monte Carlo Dropout 来近似 BNN 时，通常不需要显式计算 KL 散度损失，
    因为 Dropout 本身在训练时引入了随机性，而 MC Dropout 在推理时利用了这种随机性来估计不确定性。
    
    如果使用像 `torchbnn` 这样的库，或者自己实现了变分推断的 BNN，
    这里会计算模型权重变分后验与先验之间的 KL 散度，并将其添加到损失中以正则化。
    
    对于本项目中采用的 `nn.Dropout` 结合 MC Dropout 的简化方法，此函数返回 0。
    
    Args:
        model (nn.Module): 传入的模型 (通常是 ForwardModel，因为它包含 Dropout)。
    Returns:
        torch.Tensor: KL 散度损失 (当前为 0)。
    """
    # 由于我们使用标准的 nn.Dropout 进行 MC Dropout，没有显式的变分层，
    # 因此这里返回0。如果未来引入了变分层，需要在此处实现 KL 散度计算。
    return torch.zeros(1, device=next(model.parameters()).device)
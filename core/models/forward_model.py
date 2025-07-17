# PI_GAN_THZ/core/models/forward_model.py 

import torch
import torch.nn as nn

class ForwardModel(nn.Module):
    """
    前向仿真模型：
    输入：归一化的结构参数 (r1, r2, w, g，共 input_param_dim 维度)
    输出：预测的太赫兹传输光谱 (output_spectrum_dim 维度) 和 预测的物理性能指标 (output_metrics_dim 维度)。
    该模型用于提供物理信息约束，并通过Dropout层实现不确定性建模。
    """
    def __init__(self, input_param_dim: int, output_spectrum_dim: int, output_metrics_dim: int):
        """
        初始化前向模型。
        Args:
            input_param_dim (int): 输入结构参数的数量 (例如 4)。
            output_spectrum_dim (int): 输出光谱的数据点数量 (例如 250)。
            output_metrics_dim (int): 输出物理性能指标的数量 (例如 8)。
        """
        super(ForwardModel, self).__init__()
        self.output_spectrum_dim = output_spectrum_dim
        self.output_metrics_dim = output_metrics_dim

        # 模型的总输出维度是光谱维度和指标维度的总和
        total_output_dim = output_spectrum_dim + output_metrics_dim

        self.model = nn.Sequential(
            # 更多的隐藏层以增强模型的拟合能力
            nn.Linear(input_param_dim, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2), # 添加 Dropout 层进行不确定性建模

            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2), # 可以在多层添加Dropout

            # 输出层：同时预测光谱和物理指标
            nn.Linear(256, total_output_dim)
            # 注意：此处没有最终激活函数。
            # 这是因为光谱值 (dB) 可能是负数，而物理指标的范围也各不相同。
            # 保持线性输出能让网络学习到这些值的原始尺度，后续可以通过归一化/反归一化处理。
        )

    def forward(self, structural_params_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播。
        Args:
            structural_params_norm (torch.Tensor): 归一化的结构参数张量，形状为 (batch_size, input_param_dim)。
        Returns:
            tuple: (predicted_spectrum, predicted_metrics)
                predicted_spectrum (torch.Tensor): 预测的太赫兹光谱，形状为 (batch_size, output_spectrum_dim)。
                predicted_metrics (torch.Tensor): 预测的物理性能指标，形状为 (batch_size, output_metrics_dim)。
        """
        output = self.model(structural_params_norm)
        # 将输出分割为光谱部分和指标部分
        predicted_spectrum = output[:, :self.output_spectrum_dim]
        predicted_metrics = output[:, self.output_spectrum_dim:]
        return predicted_spectrum, predicted_metrics
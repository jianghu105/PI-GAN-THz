# PI_GAN_THZ/core/models/generator.py

import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim): # <--- 确保这里有 input_dim 和 output_dim
        """
        PI-GAN 的生成器。
        接收光谱作为输入，输出归一化的结构参数。

        Args:
            input_dim (int): 输入光谱的维度 (应为 cfg.SPECTRUM_DIM)。
            output_dim (int): 输出结构参数的维度 (应为 cfg.GENERATOR_OUTPUT_PARAM_DIM)。
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512), # 添加 BatchNorm 以提高训练稳定性
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, output_dim),
            nn.Tanh() # 假设参数被归一化到 [-1, 1]。如果归一化到 [0, 1]，请使用 nn.Sigmoid()
        )

    def forward(self, spectrum):
        # 确保输入光谱的形状适合线性层
        # 如果 spectrum 的维度超过 2 (例如: batch_size, channels, spectrum_dim)，则可能需要展平
        if spectrum.dim() > 2:
            spectrum = spectrum.view(spectrum.size(0), -1)
        return self.main(spectrum)
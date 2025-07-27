# PI_GAN_THZ/core/models/forward_model.py

import torch
import torch.nn as nn
import numpy as np

# 导入新的物理计算和指标提取辅助函数
from ..utils.physics_constraints import (
    calculate_srr_frequencies,
    extract_q_factor,
    extract_transmission
)

class ResidualBlock(nn.Module):
    """
    带缩放因子的残差块
    """
    def __init__(self, dim: int, scale_factor: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.scale_factor = scale_factor
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.scale_factor * self.layer(x)

class EnhancedForwardPINN(nn.Module):
    """
    物理增强的前向模型：
    输入：4维结构参数(r1,r2,w,g) (单位: 米)
    输出：250点透射光谱，并可选择性返回8个关键物理指标
    """
    
    def __init__(self, input_param_dim: int = 4, spectrum_dim: int = 250, fourier_dim: int = 32):
        super(PhysicsEnhancedForward, self).__init__()
        self.input_param_dim = input_param_dim
        self.spectrum_dim = spectrum_dim
        self.fourier_dim = fourier_dim
        
        # 参数特征提取模块 (4→64→128→256)
        self.param_feature_extractor = nn.Sequential(
            nn.Linear(input_param_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU()
        )
        
        # 傅里叶特征编码器
        self.B = nn.Parameter(torch.randn(16, 1) * 10, requires_grad=False)
        self.freq_encoder = nn.Sequential(
            nn.Linear(16 * 2, fourier_dim),
            nn.SiLU()
        )
        
        # 频谱生成核心
        self.spectrum_generator = nn.Sequential(
            nn.Linear(256 + fourier_dim, 512),
            nn.SiLU(),
            ResidualBlock(512, 0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
            ResidualBlock(256, 0.3),
            nn.Linear(256, 1),
            nn.Sigmoid() # 确保输出在[0,1]范围内
        )
        
    def forward(self, structural_params: torch.Tensor, return_metrics: bool = False):
        """
        前向传播
        Args:
            structural_params: 结构参数 (batch_size, 4) (单位: 米)
            return_metrics: 是否返回计算的物理指标
        Returns:
            - predicted_spectrum (torch.Tensor): (batch_size, 250)
            - (optional) metrics (torch.Tensor): (batch_size, 8)
        """
        batch_size = structural_params.shape[0]
        param_features = self.param_feature_extractor(structural_params)
        
        freq_coords = torch.linspace(0.1e12, 3.0e12, self.spectrum_dim, device=structural_params.device)
        
        # 预计算傅里叶特征以提高效率
        x = freq_coords.unsqueeze(1) # (250, 1)
        fx = torch.cat([torch.sin(2 * np.pi * x @ self.B.T), 
                       torch.cos(2 * np.pi * x @ self.B.T)], dim=1) # (250, 32)
        freq_features_encoded = self.freq_encoder(fx) # (250, fourier_dim)
        
        # 扩展特征以匹配批次大小
        param_features_expanded = param_features.unsqueeze(1).expand(-1, self.spectrum_dim, -1) # (batch, 250, 256)
        freq_features_expanded = freq_features_encoded.unsqueeze(0).expand(batch_size, -1, -1) # (batch, 250, fourier_dim)
        
        # 融合特征并生成光谱
        combined_features = torch.cat([param_features_expanded, freq_features_expanded], dim=2) # (batch, 250, 256+fourier_dim)
        predicted_spectrum = self.spectrum_generator(combined_features).squeeze(-1) # (batch, 250)

        if not return_metrics:
            return predicted_spectrum
        
        # --- 如果需要，计算并返回物理指标 ---
        r1, r2, w, g = structural_params.unbind(1)
        
        # 1. 直接计算理论f1,f2 (物理模型)
        f1_theory, f2_theory = calculate_srr_frequencies(r1, r2, w, g)
        
        # 2. 从光谱提取Q因子和透射强度
        q1 = extract_q_factor(predicted_spectrum, f1_theory, min_q=10, max_q=50)
        q2 = extract_q_factor(predicted_spectrum, f2_theory, min_q=5, max_q=30)
        s1 = extract_transmission(predicted_spectrum, f1_theory)
        s2 = extract_transmission(predicted_spectrum, f2_theory)

        # FoM 计算暂时省略，因为 calculate_fom 未定义
        fom1 = torch.zeros_like(q1) # 占位符
        fom2 = torch.zeros_like(q2) # 占位符
        
        metrics = torch.stack([
            f1_theory, q1, fom1, s1, 
            f2_theory, q2, fom2, s2
        ], dim=1)
        
        return predicted_spectrum, metrics

# PI_GAN_THZ/core/models/enhanced_forward_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnhancedForwardPINN(nn.Module):
    """
    增强版前向PINN模型：
    输入：4维结构参数(r1,r2,w,g)
    输出：250点透射光谱 + 物理约束残差
    """
    
    def __init__(self, input_param_dim: int = 4, spectrum_dim: int = 250, fourier_dim: int = 32):
        super(EnhancedForwardPINN, self).__init__()
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
        self.B = nn.Parameter(torch.randn(16, 1) * 10, requires_grad=False)  # 固定高斯矩阵
        self.freq_encoder = nn.Sequential(
            nn.Linear(16 * 2, fourier_dim),  # sin/cos各16维
            nn.SiLU()
        )
        
        # 频谱生成核心 (带残差块)
        self.spectrum_generator = nn.Sequential(
            nn.Linear(256 + fourier_dim, 512),
            nn.SiLU(),
            ResidualBlock(512, 0.3),  # 带缩放因子的残差块
            nn.Linear(512, 256),
            nn.SiLU(),
            ResidualBlock(256, 0.3),
            nn.Linear(256, spectrum_dim)
        )
        
    def forward(self, structural_params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            structural_params: 归一化的结构参数 (batch_size, 4)
        Returns:
            tuple: (predicted_spectrum, physics_residual)
        """
        batch_size = structural_params.shape[0]
        
        # 参数特征提取
        param_features = self.param_feature_extractor(structural_params)  # (batch, 256)
        
        # 生成频率坐标
        freq_coords = torch.linspace(0.5, 3.0, self.spectrum_dim).to(structural_params.device)  # (250,)
        
        # 傅里叶特征编码
        freq_features = []
        for i in range(self.spectrum_dim):
            x = freq_coords[i].view(1, 1)
            # 计算傅里叶特征
            fx = torch.cat([torch.sin(2 * np.pi * x @ self.B.T), 
                           torch.cos(2 * np.pi * x @ self.B.T)], dim=1)  # (1, 32)
            freq_features.append(self.freq_encoder(fx))  # (1, fourier_dim)
        
        freq_features = torch.cat(freq_features, dim=0)  # (250, fourier_dim)
        freq_features = freq_features.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, 250, fourier_dim)
        
        # 频谱生成
        spectra = []
        for i in range(self.spectrum_dim):
            # 融合参数特征和频率编码
            combined_features = torch.cat([
                param_features,  # (batch, 256)
                freq_features[:, i, :]  # (batch, fourier_dim)
            ], dim=1)  # (batch, 256+fourier_dim)
            
            # 生成单个频率点的光谱值
            spectrum_point = self.spectrum_generator(combined_features)  # (batch, 1)
            spectra.append(spectrum_point)
            
        predicted_spectrum = torch.cat(spectra, dim=1)  # (batch, 250)
        
        # 计算物理约束残差（仅在共振峰区域）
        physics_residual = self.compute_physics_residual(predicted_spectrum, structural_params)
        
        return predicted_spectrum, physics_residual
    
    def compute_physics_residual(self, spectrum: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """
        计算区域化物理约束残差（仅在共振峰区域）
        """
        # 简化版麦克斯韦方程残差计算
        # 找到共振峰位置（透射率极小值点）
        batch_size = spectrum.shape[0]
        residuals = []
        
        for i in range(batch_size):
            # 找到局部极小值点
            spec = spectrum[i]
            # 简化处理：假设共振峰在特定频率范围内
            # 实际应用中需要更复杂的峰值检测算法
            peak_regions = [(80, 120), (180, 220)]  # 假设两个共振峰区域
            
            region_residuals = []
            for start, end in peak_regions:
                region_spec = spec[start:end]
                # 计算二阶导数作为平滑性约束
                if len(region_spec) >= 3:
                    diff1 = region_spec[1:] - region_spec[:-1]
                    diff2 = diff1[1:] - diff1[:-1]
                    region_residual = torch.mean(diff2 ** 2)
                    region_residuals.append(region_residual)
            
            if region_residuals:
                residuals.append(torch.mean(torch.stack(region_residuals)))
            else:
                residuals.append(torch.tensor(0.0, device=spectrum.device))
        
        return torch.stack(residuals)


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
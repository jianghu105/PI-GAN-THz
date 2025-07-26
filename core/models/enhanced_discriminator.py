# PI_GAN_THZ/core/models/enhanced_discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDiscriminator(nn.Module):
    """
    增强版判别器（双头设计）：
    1. 真实/生成判别头
    2. 物理一致性判别头
    """
    
    def __init__(self, spectrum_dim: int = 250, param_dim: int = 4):
        super(EnhancedDiscriminator, self).__init__()
        self.spectrum_dim = spectrum_dim
        self.param_dim = param_dim
        
        # 共享的光谱处理分支（共振区域聚焦）
        self.spectrum_processor = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=7, padding=3),  # 较大卷积核捕获共振峰特征
            nn.LeakyReLU(0.2),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveMaxPool1d(128)  # 自适应最大池化保留关键频率点
        )
        
        # 结构参数特征提取
        self.param_processor = nn.Sequential(
            nn.Linear(param_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2)
        )
        
        # 共享特征提取（带残差块）
        self.shared_features = nn.Sequential(
            nn.Linear(48 * 128 + 32, 256),  # 光谱特征(48*128) + 参数特征(32)
            nn.LeakyReLU(0.2),
            ResidualBlockDisc(256, 0.3),  # 带缩放因子的残差连接
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2)
        )
        
        # 真实/生成判别头
        self.real_fake_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.25),  # 防止过拟合
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 物理一致性判别头
        self.physics_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spectrum: torch.Tensor, params: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            spectrum: 光谱数据 (batch_size, 250)
            params: 结构参数 (batch_size, 4)
        Returns:
            tuple: (real_fake_score, physics_score)
        """
        batch_size = spectrum.shape[0]
        
        # 光谱特征提取
        spectrum_input = spectrum.unsqueeze(1)  # (batch, 1, 250)
        spectrum_features = self.spectrum_processor(spectrum_input)  # (batch, 48, 128)
        spectrum_features = spectrum_features.view(batch_size, -1)  # (batch, 48*128)
        
        # 参数特征提取
        param_features = self.param_processor(params)  # (batch, 32)
        
        # 特征融合
        combined_features = torch.cat([spectrum_features, param_features], dim=1)  # (batch, 48*128+32)
        shared_features = self.shared_features(combined_features)  # (batch, 128)
        
        # 双头输出
        real_fake_score = self.real_fake_head(shared_features)  # (batch, 1)
        physics_score = self.physics_head(shared_features)  # (batch, 1)
        
        return real_fake_score, physics_score


class ResidualBlockDisc(nn.Module):
    """
    判别器中的残差块，带缩放因子
    """
    def __init__(self, dim: int, scale_factor: float = 0.3):
        super(ResidualBlockDisc, self).__init__()
        self.scale_factor = scale_factor
        self.layer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.scale_factor * self.layer(x)
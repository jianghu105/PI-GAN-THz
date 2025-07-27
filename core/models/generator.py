# PI_GAN_THZ/core/models/generator.py

import torch
import torch.nn as nn

# 导入新的物理约束输出层
from .physical_layers import SRRPhysicsConstrainedOutput

class EnhancedGenerator(nn.Module):
    """
    增强版生成器，集成了硬物理约束输出层。
    输入：目标光谱(250维) + 潜在向量(100维)
    输出：4维结构参数 [r1, r2, w, g] (单位: 米)，100%满足物理约束
    """
    
    def __init__(self, spectrum_dim: int = 250, z_dim: int = 100, output_dim: int = 4, hidden_dim: int = 256):
        super(EnhancedGenerator, self).__init__()
        self.spectrum_dim = spectrum_dim
        self.z_dim = z_dim
        self.output_dim = output_dim
        
        # 光谱特征提取分支 (1D CNN)
        self.spectrum_feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(64)
        )
        
        # 潜在向量处理分支
        self.latent_processor = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3)
        )
        
        # 特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 * 64 + 128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 替换为新的物理约束输出层
        self.output_layer = SRRPhysicsConstrainedOutput(hidden_dim, output_dim)
        
    def forward(self, target_spectrum: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            target_spectrum: 目标光谱 (batch_size, 250)
            latent_vector: 潜在向量 (batch_size, 100)
        Returns:
            constrained_params: 预测的、满足物理约束的结构参数 (batch_size, 4)
        """
        batch_size = target_spectrum.shape[0]
        
        # 光谱特征提取
        spectrum_input = target_spectrum.unsqueeze(1)  # (batch, 1, 250)
        spectrum_features = self.spectrum_feature_extractor(spectrum_input)
        spectrum_features = spectrum_features.view(batch_size, -1)
        
        # 潜在向量处理
        latent_features = self.latent_processor(latent_vector)
        
        # 特征融合
        combined_features = torch.cat([spectrum_features, latent_features], dim=1)
        fused_features = self.feature_fusion(combined_features)
        
        # 通过物理约束层生成最终参数
        constrained_params = self.output_layer(fused_features)
        
        return constrained_params

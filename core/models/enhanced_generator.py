# PI_GAN_THZ/core/models/enhanced_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedGenerator(nn.Module):
    """
    增强版生成器：
    输入：目标光谱(250维) + 潜在向量(100维)
    输出：4维结构参数(r1,r2,w,g)
    
    """
    
    def __init__(self, spectrum_dim: int = 250, z_dim: int = 100, output_dim: int = 4):
        super(EnhancedGenerator, self).__init__()
        self.spectrum_dim = spectrum_dim
        self.z_dim = z_dim
        self.output_dim = output_dim
        
        # 光谱特征提取分支 (1D CNN)
        self.spectrum_feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # 匹配共振峰宽度
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(64)  # 降低维度并聚焦关键特征
        )
        
        # 潜在向量处理分支
        self.latent_processor = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)  # 防止小数据集过拟合
        )
        
        # 特征融合模块
        self.feature_fusion = nn.Sequential(
            nn.Linear(32 * 64 + 128, 256),  # 32*64是光谱特征维度，128是潜在向量维度
            nn.LayerNorm(256),  # LayerNorm替代BatchNorm
            nn.LeakyReLU(0.2),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2)
        )
        
        # 物理约束输出层
        self.physical_output = nn.Linear(256, output_dim)
        
    def forward(self, target_spectrum: torch.Tensor, latent_vector: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            target_spectrum: 目标光谱 (batch_size, 250)
            latent_vector: 潜在向量 (batch_size, 100)
        Returns:
            predicted_params: 预测的结构参数 (batch_size, 4)
        """
        batch_size = target_spectrum.shape[0]
        
        # 光谱特征提取
        spectrum_input = target_spectrum.unsqueeze(1)  # (batch, 1, 250)
        spectrum_features = self.spectrum_feature_extractor(spectrum_input)  # (batch, 32, 64)
        spectrum_features = spectrum_features.view(batch_size, -1)  # (batch, 32*64)
        
        # 潜在向量处理
        latent_features = self.latent_processor(latent_vector)  # (batch, 128)
        
        # 特征融合
        combined_features = torch.cat([spectrum_features, latent_features], dim=1)  # (batch, 32*64+128)
        fused_features = self.feature_fusion(combined_features)  # (batch, 256)
        
        # 物理约束输出
        raw_params = self.physical_output(fused_features)  # (batch, 4)
        
        # 应用物理边界约束
        constrained_params = self.apply_physical_constraints(raw_params)
        
        return constrained_params
    
    def apply_physical_constraints(self, raw_params: torch.Tensor) -> torch.Tensor:
        """
        应用硬编码物理边界约束
        """
        # r1: 0.5-5μm (sigmoid约束)
        r1 = 0.5 + 4.5 * torch.sigmoid(raw_params[:, 0])
        
        # r2: <0.9×r1 (确保r2<r1)
        r2_raw = torch.sigmoid(raw_params[:, 1])
        r2 = r2_raw * 0.9 * r1
        
        # w, g: >0 (softplus约束)
        w = F.softplus(raw_params[:, 2])
        g = F.softplus(raw_params[:, 3])
        
        return torch.stack([r1, r2, w, g], dim=1)
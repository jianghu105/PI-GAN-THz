# PI_GAN_THZ/core/models/enhanced_discriminator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDiscriminator(nn.Module):
    """
    增强版判别器：分别处理光谱和参数特征，然后融合
    """
    def __init__(self, input_spec_dim, input_param_dim, use_spectral_norm=True):
        super(EnhancedDiscriminator, self).__init__()
        
        # 光谱特征提取器
        self.spectrum_encoder = nn.Sequential(
            nn.Linear(input_spec_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )
        
        # 参数特征提取器
        self.param_encoder = nn.Sequential(
            nn.Linear(input_param_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )
        
        # 融合层
        fusion_dim = 128 + 32  # spectrum features + param features
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 应用谱归一化提高训练稳定性
        if use_spectral_norm:
            self._apply_spectral_norm()
            
    def _apply_spectral_norm(self):
        """应用谱归一化到所有线性层"""
        from torch.nn.utils import spectral_norm
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                spectral_norm(module)
                
    def forward(self, spectrum, params):
        # 确保输入是2D的
        if spectrum.dim() > 2:
            spectrum = spectrum.view(spectrum.size(0), -1)
        if params.dim() > 2:
            params = params.view(params.size(0), -1)
            
        # 分别提取特征
        spectrum_features = self.spectrum_encoder(spectrum)
        param_features = self.param_encoder(params)
        
        # 特征融合
        combined_features = torch.cat([spectrum_features, param_features], dim=1)
        
        # 最终分类
        output = self.fusion_layers(combined_features)
        
        return output

class ConvDiscriminator(nn.Module):
    """
    基于卷积的判别器：更适合处理光谱数据
    """
    def __init__(self, input_spec_dim, input_param_dim):
        super(ConvDiscriminator, self).__init__()
        
        # 光谱的1D卷积处理
        self.spectrum_conv = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(16),
        )
        
        # 参数处理
        self.param_encoder = nn.Sequential(
            nn.Linear(input_param_dim, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )
        
        # 融合和分类
        conv_output_dim = 256 * 16  # 256 channels * 16 length
        self.classifier = nn.Sequential(
            nn.Linear(conv_output_dim + 32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spectrum, params):
        batch_size = spectrum.size(0)
        
        # 处理光谱数据
        if spectrum.dim() > 2:
            spectrum = spectrum.view(batch_size, -1)
        spectrum_conv_input = spectrum.unsqueeze(1)  # 添加channel维度
        spectrum_features = self.spectrum_conv(spectrum_conv_input)
        spectrum_features = spectrum_features.view(batch_size, -1)
        
        # 处理参数数据
        if params.dim() > 2:
            params = params.view(batch_size, -1)
        param_features = self.param_encoder(params)
        
        # 融合特征
        combined_features = torch.cat([spectrum_features, param_features], dim=1)
        
        # 分类
        output = self.classifier(combined_features)
        
        return output

class MultiScaleDiscriminator(nn.Module):
    """
    多尺度判别器：在不同尺度上判别，提高判别能力
    """
    def __init__(self, input_spec_dim, input_param_dim):
        super(MultiScaleDiscriminator, self).__init__()
        
        # 全尺度判别器
        self.full_scale_disc = EnhancedDiscriminator(input_spec_dim, input_param_dim)
        
        # 半尺度判别器
        self.half_scale_disc = EnhancedDiscriminator(input_spec_dim // 2, input_param_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(2, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, spectrum, params):
        batch_size = spectrum.size(0)
        
        # 全尺度判别
        full_output = self.full_scale_disc(spectrum, params)
        
        # 半尺度判别（下采样）
        half_spectrum = F.avg_pool1d(spectrum.unsqueeze(1), kernel_size=2).squeeze(1)
        half_output = self.half_scale_disc(half_spectrum, params)
        
        # 融合多尺度结果
        combined = torch.cat([full_output, half_output], dim=1)
        final_output = self.fusion(combined)
        
        return final_output
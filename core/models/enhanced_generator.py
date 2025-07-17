# PI_GAN_THZ/core/models/enhanced_generator.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedGenerator(nn.Module):
    """
    增强版生成器：结合1D卷积和注意力机制处理光谱数据
    """
    def __init__(self, input_dim, output_dim, use_attention=True):
        super(EnhancedGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # 1D卷积层：提取光谱特征
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv1d(1, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # 第二个卷积块
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            # 第三个卷积块
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(32),  # 固定输出长度
        )
        
        # 计算卷积后的特征维度
        conv_output_dim = 256 * 32  # 256 channels * 32 length
        
        # 注意力机制
        if self.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=256, 
                num_heads=8, 
                dropout=0.1,
                batch_first=True
            )
            
        # 全连接层：深度特征提取
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_output_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim),
            nn.Tanh()  # 输出归一化到[-1, 1]
        )
        
    def forward(self, spectrum):
        batch_size = spectrum.size(0)
        
        # 确保输入是2D的
        if spectrum.dim() > 2:
            spectrum = spectrum.view(batch_size, -1)
            
        # 为1D卷积添加channel维度
        x = spectrum.unsqueeze(1)  # (batch_size, 1, spectrum_length)
        
        # 1D卷积特征提取
        conv_features = self.conv_layers(x)  # (batch_size, 256, 32)
        
        # 注意力机制
        if self.use_attention:
            # 调整维度用于注意力计算
            attn_input = conv_features.permute(0, 2, 1)  # (batch_size, 32, 256)
            attn_output, _ = self.attention(attn_input, attn_input, attn_input)
            conv_features = attn_output.permute(0, 2, 1)  # (batch_size, 256, 32)
        
        # 展平特征
        flattened = conv_features.view(batch_size, -1)
        
        # 全连接层处理
        output = self.fc_layers(flattened)
        
        return output

class ResidualBlock(nn.Module):
    """残差块：增强网络表达能力"""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class ResidualGenerator(nn.Module):
    """
    基于残差连接的生成器
    """
    def __init__(self, input_dim, output_dim, num_residual_blocks=3):
        super(ResidualGenerator, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True)
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(512) for _ in range(num_residual_blocks)
        ])
        
        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_dim),
            nn.Tanh()
        )
        
    def forward(self, spectrum):
        if spectrum.dim() > 2:
            spectrum = spectrum.view(spectrum.size(0), -1)
            
        x = self.input_projection(spectrum)
        
        for block in self.residual_blocks:
            x = block(x)
            
        output = self.output_layers(x)
        return output
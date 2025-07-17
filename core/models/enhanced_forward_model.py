# PI_GAN_THZ/core/models/enhanced_forward_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedForwardModel(nn.Module):
    """
    增强版前向模型：使用多分支架构分别处理光谱和物理指标预测
    """
    def __init__(self, input_param_dim, output_spectrum_dim, output_metrics_dim):
        super(EnhancedForwardModel, self).__init__()
        
        self.output_spectrum_dim = output_spectrum_dim
        self.output_metrics_dim = output_metrics_dim
        
        # 共享特征提取器
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_param_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # 光谱预测分支
        self.spectrum_branch = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(1024, output_spectrum_dim),
            # 不使用激活函数，因为光谱值可能为负
        )
        
        # 物理指标预测分支
        self.metrics_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            nn.Linear(64, output_metrics_dim),
            # 不使用激活函数，保持原始输出范围
        )
        
    def forward(self, structural_params_norm):
        # 共享特征提取
        shared_features = self.shared_encoder(structural_params_norm)
        
        # 分支预测
        predicted_spectrum = self.spectrum_branch(shared_features)
        predicted_metrics = self.metrics_branch(shared_features)
        
        return predicted_spectrum, predicted_metrics

class PhysicsInformedForwardModel(nn.Module):
    """
    物理信息前向模型：集成物理约束的前向模型
    """
    def __init__(self, input_param_dim, output_spectrum_dim, output_metrics_dim):
        super(PhysicsInformedForwardModel, self).__init__()
        
        self.output_spectrum_dim = output_spectrum_dim
        self.output_metrics_dim = output_metrics_dim
        
        # 参数嵌入层
        self.param_embedding = nn.Sequential(
            nn.Linear(input_param_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
        )
        
        # 物理约束层：编码物理规律
        self.physics_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # 光谱生成器：基于物理原理生成光谱
        self.spectrum_generator = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(1024, output_spectrum_dim),
        )
        
        # 指标预测器：基于物理参数预测性能指标
        self.metrics_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(128, output_metrics_dim),
        )
        
        # 注意力机制：加权融合不同物理特征
        self.attention = nn.MultiheadAttention(
            embed_dim=512,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
    def forward(self, structural_params_norm):
        batch_size = structural_params_norm.size(0)
        
        # 参数嵌入
        param_embed = self.param_embedding(structural_params_norm)
        
        # 物理约束编码
        physics_features = self.physics_encoder(param_embed)
        
        # 自注意力机制
        attn_input = physics_features.unsqueeze(1)  # (batch_size, 1, 512)
        attn_output, _ = self.attention(attn_input, attn_input, attn_input)
        enhanced_features = attn_output.squeeze(1)  # (batch_size, 512)
        
        # 分别预测光谱和指标
        predicted_spectrum = self.spectrum_generator(enhanced_features)
        predicted_metrics = self.metrics_predictor(enhanced_features)
        
        return predicted_spectrum, predicted_metrics

class UncertaintyForwardModel(nn.Module):
    """
    不确定性前向模型：显式建模预测不确定性
    """
    def __init__(self, input_param_dim, output_spectrum_dim, output_metrics_dim):
        super(UncertaintyForwardModel, self).__init__()
        
        self.output_spectrum_dim = output_spectrum_dim
        self.output_metrics_dim = output_metrics_dim
        
        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_param_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        
        # 光谱均值预测
        self.spectrum_mean = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, output_spectrum_dim),
        )
        
        # 光谱方差预测
        self.spectrum_var = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, output_spectrum_dim),
            nn.Softplus()  # 确保方差为正
        )
        
        # 指标均值预测
        self.metrics_mean = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, output_metrics_dim),
        )
        
        # 指标方差预测
        self.metrics_var = nn.Sequential(
            nn.Linear(1024, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, output_metrics_dim),
            nn.Softplus()  # 确保方差为正
        )
        
    def forward(self, structural_params_norm):
        # 提取特征
        features = self.feature_extractor(structural_params_norm)
        
        # 预测均值和方差
        spectrum_mean = self.spectrum_mean(features)
        spectrum_var = self.spectrum_var(features)
        
        metrics_mean = self.metrics_mean(features)
        metrics_var = self.metrics_var(features)
        
        # 在训练时返回均值，在推理时可以采样
        if self.training:
            return spectrum_mean, metrics_mean
        else:
            # 可以根据需要返回不确定性信息
            return spectrum_mean, metrics_mean, spectrum_var, metrics_var

    def sample_predictions(self, structural_params_norm, num_samples=100):
        """
        使用预测的均值和方差进行蒙特卡洛采样
        """
        features = self.feature_extractor(structural_params_norm)
        
        spectrum_mean = self.spectrum_mean(features)
        spectrum_var = self.spectrum_var(features)
        
        metrics_mean = self.metrics_mean(features)
        metrics_var = self.metrics_var(features)
        
        # 蒙特卡洛采样
        spectrum_samples = []
        metrics_samples = []
        
        for _ in range(num_samples):
            spectrum_sample = torch.normal(spectrum_mean, spectrum_var.sqrt())
            metrics_sample = torch.normal(metrics_mean, metrics_var.sqrt())
            
            spectrum_samples.append(spectrum_sample)
            metrics_samples.append(metrics_sample)
        
        return torch.stack(spectrum_samples), torch.stack(metrics_samples)
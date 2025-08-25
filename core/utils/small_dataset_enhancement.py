"""
小数据集增强策略模块
专门为4450条数据的小数据集设计的数据增强和训练策略
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler
import random
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import logging

logger = logging.getLogger(__name__)

class SpectralDataAugmenter:
    """光谱数据增强器，专门针对太赫兹光谱数据"""
    
    def __init__(self, noise_level: float = 0.02, shift_range: float = 0.1):
        """
        Args:
            noise_level: 噪声水平 (相对于信号强度)
            shift_range: 频率偏移范围 (相对于频率范围)
        """
        self.noise_level = noise_level
        self.shift_range = shift_range
        
    def add_gaussian_noise(self, spectra: torch.Tensor) -> torch.Tensor:
        """添加高斯噪声"""
        noise = torch.randn_like(spectra) * self.noise_level * spectra.std()
        return spectra + noise
    
    def frequency_shift(self, spectra: torch.Tensor) -> torch.Tensor:
        """频率偏移增强"""
        batch_size, freq_points = spectra.shape
        augmented = torch.zeros_like(spectra)
        
        for i in range(batch_size):
            # 随机偏移量
            shift_pixels = int(random.uniform(-self.shift_range, self.shift_range) * freq_points)
            
            if shift_pixels > 0:
                augmented[i, shift_pixels:] = spectra[i, :-shift_pixels]
                # 边界填充
                augmented[i, :shift_pixels] = spectra[i, 0]
            elif shift_pixels < 0:
                augmented[i, :shift_pixels] = spectra[i, -shift_pixels:]
                # 边界填充
                augmented[i, shift_pixels:] = spectra[i, -1]
            else:
                augmented[i] = spectra[i]
                
        return augmented
    
    def smoothing_variation(self, spectra: torch.Tensor, window_length: int = 5) -> torch.Tensor:
        """平滑变化增强"""
        augmented = spectra.clone()
        
        for i in range(spectra.shape[0]):
            spectrum = spectra[i].numpy()
            if len(spectrum) > window_length:
                # 应用Savitzky-Golay滤波器
                smoothed = savgol_filter(spectrum, window_length, 3)
                augmented[i] = torch.from_numpy(smoothed).float()
        
        return augmented
    
    def amplitude_scaling(self, spectra: torch.Tensor, scale_range: float = 0.1) -> torch.Tensor:
        """幅度缩放增强"""
        scale_factors = 1 + torch.randn(spectra.shape[0], 1) * scale_range
        return spectra * scale_factors
    
    def apply_random_augmentation(self, spectra: torch.Tensor, 
                                struct_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        """随机应用增强技术"""
        augmentation_methods = [
            self.add_gaussian_noise,
            self.frequency_shift,
            self.smoothing_variation,
            self.amplitude_scaling
        ]
        
        # 随机选择1-2种增强方法
        num_methods = random.randint(1, 2)
        selected_methods = random.sample(augmentation_methods, num_methods)
        
        augmented_spectra = spectra.clone()
        for method in selected_methods:
            augmented_spectra = method(augmented_spectra)
            
        return augmented_spectra

class StructuralParameterAugmenter:
    """结构参数增强器"""
    
    def __init__(self, noise_level: float = 0.01, bounds: Optional[Dict] = None):
        self.noise_level = noise_level
        self.bounds = bounds or {}
        
    def add_parameter_noise(self, struct_params: torch.Tensor) -> torch.Tensor:
        """为结构参数添加小幅度噪声"""
        noise = torch.randn_like(struct_params) * self.noise_level
        augmented = struct_params + noise
        
        # 确保在合理范围内
        if self.bounds:
            for i, param_name in enumerate(['r1', 'r2', 'w', 'g']):
                if param_name in self.bounds:
                    min_val, max_val = self.bounds[param_name]
                    augmented[:, i] = torch.clamp(augmented[:, i], min_val, max_val)
        
        return augmented
    
    def interpolate_parameters(self, struct_params: torch.Tensor) -> torch.Tensor:
        """参数插值增强"""
        batch_size = struct_params.shape[0]
        if batch_size < 2:
            return struct_params
            
        # 随机选择两个样本进行插值
        indices = torch.randperm(batch_size)[:2]
        alpha = torch.rand(1).item()
        
        interpolated = alpha * struct_params[indices[0]] + (1 - alpha) * struct_params[indices[1]]
        
        # 随机替换一些样本
        replace_indices = torch.randperm(batch_size)[:batch_size//4]
        augmented = struct_params.clone()
        for idx in replace_indices:
            augmented[idx] = interpolated
            
        return augmented

class SmallDatasetTrainingStrategy:
    """小数据集训练策略"""
    
    def __init__(self, total_samples: int = 4450):
        self.total_samples = total_samples
        self.logger = logging.getLogger(__name__)
        
    def get_regularization_config(self) -> Dict[str, float]:
        """获取针对小数据集的正则化配置"""
        # 小数据集需要更强的正则化
        return {
            'dropout_rate': 0.3,  # 增加dropout
            'weight_decay': 1e-3,  # 增加权重衰减
            'batch_norm_momentum': 0.9,  # 调整批归一化动量
            'gradient_clip_norm': 1.0,  # 梯度裁剪
            'label_smoothing': 0.1,  # 标签平滑
        }
    
    def get_training_schedule(self) -> Dict[str, any]:
        """获取训练计划"""
        return {
            'warmup_epochs': 50,  # 预热期
            'main_epochs': 400,   # 主要训练期
            'lr_decay_patience': 20,  # 学习率衰减耐心值
            'early_stopping_patience': 50,  # 早停耐心值
            'lr_decay_factor': 0.5,  # 学习率衰减因子
            'min_lr': 1e-6,  # 最小学习率
        }
    
    def get_augmentation_strategy(self) -> Dict[str, any]:
        """获取数据增强策略"""
        return {
            'augmentation_ratio': 0.5,  # 50%的数据进行增强
            'spectral_noise_level': 0.02,
            'structural_noise_level': 0.01,
            'augmentation_per_epoch': True,  # 每个epoch都进行增强
        }

class CurriculumLearning:
    """课程学习策略，从简单到复杂"""
    
    def __init__(self, total_epochs: int = 500):
        self.total_epochs = total_epochs
        
    def get_difficulty_schedule(self, current_epoch: int) -> Dict[str, float]:
        """获取当前epoch的难度设置"""
        progress = current_epoch / self.total_epochs
        
        # 逐步增加训练难度
        return {
            'noise_level': 0.01 + 0.02 * progress,  # 噪声从小到大
            'augmentation_strength': 0.3 + 0.7 * progress,  # 增强强度递增
            'loss_weights': {
                'physics_weight': 0.05 + 0.15 * progress,  # 物理损失权重递增
                'diversity_weight': 0.05 * progress,  # 多样性损失后期引入
            }
        }

class ModelEnsemble:
    """模型集成策略，针对小数据集"""
    
    def __init__(self, n_models: int = 3):
        self.n_models = n_models
        self.models = []
        
    def create_diverse_models(self, base_model_class, model_kwargs: Dict):
        """创建多样化的模型"""
        configs = [
            {'hidden_dims': [128, 64, 32]},  # 小模型
            {'hidden_dims': [256, 128, 64]},  # 中等模型
            {'hidden_dims': [512, 256, 128, 64]},  # 大模型
        ]
        
        for i in range(self.n_models):
            config = configs[i % len(configs)]
            model_config = {**model_kwargs, **config}
            model = base_model_class(**model_config)
            self.models.append(model)
            
        return self.models
    
    def ensemble_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """集成预测"""
        predictions = []
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(inputs)
                predictions.append(pred)
        
        # 平均预测结果
        return torch.stack(predictions).mean(dim=0)

class GradientAccumulation:
    """梯度累积，模拟更大的批次"""
    
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        
    def should_update(self) -> bool:
        """判断是否应该更新参数"""
        self.current_step += 1
        if self.current_step >= self.accumulation_steps:
            self.current_step = 0
            return True
        return False
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """缩放损失以配合梯度累积"""
        return loss / self.accumulation_steps

def create_small_dataset_training_pipeline(model, train_loader, val_loader, 
                                         device: str = 'cuda') -> Dict[str, any]:
    """创建完整的小数据集训练管道"""
    
    strategy = SmallDatasetTrainingStrategy(total_samples=4450)
    
    # 正则化配置
    reg_config = strategy.get_regularization_config()
    
    # 训练计划
    schedule = strategy.get_training_schedule()
    
    # 数据增强
    spectral_augmenter = SpectralDataAugmenter(
        noise_level=0.02,
        shift_range=0.1
    )
    
    structural_augmenter = StructuralParameterAugmenter(
        noise_level=0.01
    )
    
    # 课程学习
    curriculum = CurriculumLearning(total_epochs=schedule['main_epochs'])
    
    # 梯度累积
    grad_accumulation = GradientAccumulation(accumulation_steps=4)
    
    return {
        'model': model,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'device': device,
        'regularization': reg_config,
        'schedule': schedule,
        'spectral_augmenter': spectral_augmenter,
        'structural_augmenter': structural_augmenter,
        'curriculum': curriculum,
        'grad_accumulation': grad_accumulation,
    }

if __name__ == '__main__':
    # 测试数据增强
    print("=== 测试小数据集增强策略 ===")
    
    # 模拟数据
    batch_size = 32
    spectra_dim = 250
    struct_dim = 4
    
    sample_spectra = torch.randn(batch_size, spectra_dim)
    sample_struct = torch.randn(batch_size, struct_dim)
    
    # 光谱增强测试
    spectral_aug = SpectralDataAugmenter()
    augmented_spectra = spectral_aug.apply_random_augmentation(sample_spectra)
    print(f"原始光谱形状: {sample_spectra.shape}")
    print(f"增强后光谱形状: {augmented_spectra.shape}")
    print(f"光谱变化幅度: {(augmented_spectra - sample_spectra).abs().mean():.6f}")
    
    # 结构参数增强测试
    struct_aug = StructuralParameterAugmenter()
    augmented_struct = struct_aug.add_parameter_noise(sample_struct)
    print(f"原始结构参数形状: {sample_struct.shape}")
    print(f"增强后结构参数形状: {augmented_struct.shape}")
    print(f"结构参数变化幅度: {(augmented_struct - sample_struct).abs().mean():.6f}")
    
    # 训练策略测试
    strategy = SmallDatasetTrainingStrategy()
    reg_config = strategy.get_regularization_config()
    print(f"正则化配置: {reg_config}")
    
    schedule = strategy.get_training_schedule()
    print(f"训练计划: {schedule}")
    
    # 课程学习测试
    curriculum = CurriculumLearning()
    difficulty = curriculum.get_difficulty_schedule(100)  # 第100个epoch
    print(f"第100epoch难度设置: {difficulty}")
    
    print("小数据集增强策略测试完成！")
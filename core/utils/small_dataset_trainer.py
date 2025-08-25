"""
小数据集专用训练器
集成数据增强、交叉验证、正则化等技术
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from .small_dataset_enhancement import (
    SpectralDataAugmenter, 
    StructuralParameterAugmenter,
    SmallDatasetTrainingStrategy,
    CurriculumLearning,
    GradientAccumulation
)

logger = logging.getLogger(__name__)

class SmallDatasetTrainer:
    """专门针对小数据集的训练器"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        
        # 初始化策略组件
        self.strategy = SmallDatasetTrainingStrategy(total_samples=4450)
        self.spectral_augmenter = SpectralDataAugmenter(noise_level=0.02, shift_range=0.1)
        self.structural_augmenter = StructuralParameterAugmenter(noise_level=0.01)
        self.curriculum = CurriculumLearning(total_epochs=500)
        self.grad_accumulation = GradientAccumulation(accumulation_steps=4)
        
        # 训练历史记录
        self.train_history = defaultdict(list)
        self.val_history = defaultdict(list)
        
        # 最佳模型状态
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
        
        # 配置
        self.reg_config = self.strategy.get_regularization_config()
        self.schedule = self.strategy.get_training_schedule()
        
    def setup_optimizer_and_scheduler(self, learning_rate: float = 1e-4):
        """设置优化器和学习率调度器"""
        # 使用AdamW优化器，包含权重衰减
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=self.reg_config['weight_decay'],
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.schedule['lr_decay_factor'],
            patience=self.schedule['lr_decay_patience'],
            min_lr=self.schedule['min_lr'],
            verbose=True
        )
        
        # 预热调度器
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.schedule['warmup_epochs']
        )
        
    def apply_data_augmentation(self, batch: Dict[str, torch.Tensor], 
                              augmentation_ratio: float = 0.5) -> Dict[str, torch.Tensor]:
        """应用数据增强"""
        batch_size = batch['struct'].shape[0]
        augment_count = int(batch_size * augmentation_ratio)
        
        if augment_count == 0:
            return batch
        
        # 随机选择要增强的样本
        augment_indices = torch.randperm(batch_size)[:augment_count]
        
        augmented_batch = {
            'struct': batch['struct'].clone(),
            'spectra': batch['spectra'].clone(),
            'metrics': batch['metrics'].clone()
        }
        
        # 应用光谱增强
        if len(augment_indices) > 0:
            augmented_spectra = self.spectral_augmenter.apply_random_augmentation(
                batch['spectra'][augment_indices]
            )
            augmented_batch['spectra'][augment_indices] = augmented_spectra
            
            # 应用结构参数增强
            augmented_struct = self.structural_augmenter.add_parameter_noise(
                batch['struct'][augment_indices]
            )
            augmented_batch['struct'][augment_indices] = augmented_struct
        
        return augmented_batch
    
    def compute_loss_with_regularization(self, predictions: torch.Tensor, 
                                       targets: torch.Tensor, 
                                       model: nn.Module) -> torch.Tensor:
        """计算包含正则化的损失"""
        # 基础损失
        base_loss = nn.MSELoss()(predictions, targets)
        
        # L2正则化（权重衰减已在优化器中处理）
        l2_reg = 0.0
        for param in model.parameters():
            l2_reg += torch.norm(param, p=2)
        
        # 标签平滑（如果适用）
        if hasattr(self.reg_config, 'label_smoothing') and self.reg_config['label_smoothing'] > 0:
            # 对于回归任务，标签平滑可以通过添加小噪声实现
            noise = torch.randn_like(targets) * self.reg_config['label_smoothing'] * targets.std()
            smoothed_targets = targets + noise
            smooth_loss = nn.MSELoss()(predictions, smoothed_targets)
            base_loss = 0.9 * base_loss + 0.1 * smooth_loss
        
        total_loss = base_loss + self.reg_config['weight_decay'] * l2_reg
        
        return total_loss
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        epoch_losses = []
        
        # 获取当前epoch的课程学习设置
        difficulty = self.curriculum.get_difficulty_schedule(epoch)
        
        for batch_idx, batch in enumerate(train_loader):
            # 移动数据到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 应用数据增强
            if epoch > self.schedule['warmup_epochs']:  # 预热期后开始增强
                batch = self.apply_data_augmentation(batch, 
                                                   difficulty['augmentation_strength'])
            
            # 前向传播
            if hasattr(self.model, 'forward'):
                predictions = self.model(batch['struct'])
            else:
                predictions = self.model(batch['struct'])
            
            # 计算损失
            if isinstance(predictions, tuple):
                # 如果模型返回多个输出（如前向模型）
                pred_spectra, pred_metrics = predictions
                loss_spectra = self.compute_loss_with_regularization(
                    pred_spectra, batch['spectra'], self.model
                )
                loss_metrics = self.compute_loss_with_regularization(
                    pred_metrics, batch['metrics'], self.model
                )
                loss = config.SPECTRA_LOSS_WEIGHT * loss_spectra + config.METRIC_LOSS_WEIGHT * loss_metrics
            else:
                # 单一输出
                loss = self.compute_loss_with_regularization(
                    predictions, batch['spectra'], self.model
                )
            
            # 梯度累积
            loss = self.grad_accumulation.scale_loss(loss)
            loss.backward()
            
            # 梯度裁剪
            if self.reg_config['gradient_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.reg_config['gradient_clip_norm']
                )
            
            # 更新参数
            if self.grad_accumulation.should_update():
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            epoch_losses.append(loss.item() * self.grad_accumulation.accumulation_steps)
        
        # 更新学习率（预热期）
        if epoch < self.schedule['warmup_epochs']:
            self.warmup_scheduler.step()
        
        return {'train_loss': np.mean(epoch_losses)}
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """验证一个epoch"""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                predictions = self.model(batch['struct'])
                
                # 计算损失
                if isinstance(predictions, tuple):
                    pred_spectra, pred_metrics = predictions
                    loss_spectra = nn.MSELoss()(pred_spectra, batch['spectra'])
                    loss_metrics = nn.MSELoss()(pred_metrics, batch['metrics'])
                    loss = config.SPECTRA_LOSS_WEIGHT * loss_spectra + config.METRIC_LOSS_WEIGHT * loss_metrics
                else:
                    loss = nn.MSELoss()(predictions, batch['spectra'])
                
                val_losses.append(loss.item())
        
        return {'val_loss': np.mean(val_losses)}
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = None, save_path: str = None) -> Dict[str, List[float]]:
        """完整的训练流程"""
        if epochs is None:
            epochs = self.schedule['main_epochs']
        
        logger.info(f"开始小数据集训练，总共 {epochs} 个epoch")
        logger.info(f"正则化配置: {self.reg_config}")
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # 验证
            val_metrics = self.validate_epoch(val_loader)
            
            # 记录历史
            for key, value in train_metrics.items():
                self.train_history[key].append(value)
            for key, value in val_metrics.items():
                self.val_history[key].append(value)
            
            # 学习率调度
            if epoch >= self.schedule['warmup_epochs']:
                self.scheduler.step(val_metrics['val_loss'])
            
            # 早停检查
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_model_state = self.model.state_dict().copy()
                self.patience_counter = 0
                
                # 保存最佳模型
                if save_path:
                    torch.save(self.best_model_state, save_path)
                    logger.info(f"保存最佳模型到 {save_path}")
            else:
                self.patience_counter += 1
            
            # 打印进度
            if epoch % 10 == 0 or epoch == epochs - 1:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_metrics['train_loss']:.6f}, "
                    f"Val Loss: {val_metrics['val_loss']:.6f}, "
                    f"LR: {current_lr:.2e}, "
                    f"Patience: {self.patience_counter}/{self.schedule['early_stopping_patience']}"
                )
            
            # 早停
            if self.patience_counter >= self.schedule['early_stopping_patience']:
                logger.info(f"早停触发，在第 {epoch+1} epoch停止训练")
                break
        
        # 恢复最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info("恢复最佳模型状态")
        
        return {
            'train_history': dict(self.train_history),
            'val_history': dict(self.val_history),
            'best_val_loss': self.best_val_loss
        }
    
    def cross_validate(self, dataset, n_folds: int = 5, 
                      epochs_per_fold: int = 100) -> Dict[str, Any]:
        """K折交叉验证"""
        logger.info(f"开始 {n_folds} 折交叉验证")
        
        # 准备数据
        all_data = {
            'struct': torch.cat([batch['struct'] for batch in dataset]),
            'spectra': torch.cat([batch['spectra'] for batch in dataset]),
            'metrics': torch.cat([batch['metrics'] for batch in dataset])
        }
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(all_data['struct'])):
            logger.info(f"训练第 {fold+1}/{n_folds} 折")
            
            # 创建折数据
            train_data = {k: v[train_idx] for k, v in all_data.items()}
            val_data = {k: v[val_idx] for k, v in all_data.items()}
            
            # 创建数据加载器
            from ..utils.data_loader import MetamaterialDataset
            train_dataset = MetamaterialDataset(
                train_data['struct'], train_data['spectra'], train_data['metrics']
            )
            val_dataset = MetamaterialDataset(
                val_data['struct'], val_data['spectra'], val_data['metrics']
            )
            
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                    shuffle=True, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, 
                                  shuffle=False)
            
            # 重置模型和优化器
            self._reset_model()
            self.setup_optimizer_and_scheduler()
            
            # 训练当前折
            fold_history = self.train(train_loader, val_loader, epochs_per_fold)
            fold_results.append({
                'fold': fold + 1,
                'best_val_loss': fold_history['best_val_loss'],
                'final_train_loss': fold_history['train_history']['train_loss'][-1],
                'history': fold_history
            })
        
        # 汇总结果
        avg_val_loss = np.mean([result['best_val_loss'] for result in fold_results])
        std_val_loss = np.std([result['best_val_loss'] for result in fold_results])
        
        logger.info(f"交叉验证完成 - 平均验证损失: {avg_val_loss:.6f} ± {std_val_loss:.6f}")
        
        return {
            'fold_results': fold_results,
            'average_val_loss': avg_val_loss,
            'std_val_loss': std_val_loss,
            'best_fold': min(fold_results, key=lambda x: x['best_val_loss'])
        }
    
    def _reset_model(self):
        """重置模型参数"""
        for layer in self.model.modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """绘制训练历史"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        axes[0].plot(self.train_history['train_loss'], label='Training Loss')
        axes[0].plot(self.val_history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training History')
        axes[0].legend()
        axes[0].grid(True)
        
        # 学习率曲线
        if hasattr(self, 'lr_history'):
            axes[1].plot(self.lr_history, label='Learning Rate')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_title('Learning Rate Schedule')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"训练历史图保存到 {save_path}")
        
        plt.show()

if __name__ == '__main__':
    print("=== 小数据集训练器测试 ===")
    
    # 这里应该加载实际的模型和数据进行测试
    # 由于依赖关系，这里只进行基本的初始化测试
    
    # 模拟模型
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 250)
        
        def forward(self, x):
            return self.fc(x)
    
    model = MockModel()
    trainer = SmallDatasetTrainer(model, device='cpu')
    
    print(f"正则化配置: {trainer.reg_config}")
    print(f"训练计划: {trainer.schedule}")
    print("小数据集训练器初始化成功！")
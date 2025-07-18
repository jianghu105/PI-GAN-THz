# PI_GAN_THZ/core/train/unified_trainer.py
# 统一训练器 - 整合所有训练功能

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.models.forward_model import ForwardModel
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_bce, criterion_mse
import config.config as cfg
from config.training_optimization import get_optimization_config

class UnifiedTrainer:
    """
    统一训练器 - 整合前向模型预训练、PI-GAN训练和优化训练
    """
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.opt_config = get_optimization_config()
        
        # 模型
        self.generator = None
        self.discriminator = None
        self.forward_model = None
        
        # 优化器
        self.optimizer_G = None
        self.optimizer_D = None
        self.optimizer_F = None
        
        # 调度器
        self.scheduler_G = None
        self.scheduler_D = None
        self.scheduler_F = None
        
        # 损失函数
        self.criterion_bce = criterion_bce()
        self.criterion_mse = criterion_mse()
        
        # 训练历史
        self.train_history = {
            'forward_losses': [],      # 前向模型训练损失
            'g_losses': [],           # 生成器损失
            'd_losses': [],           # 判别器损失
            'constraint_violations': [], # 约束违约率
            'detailed_losses': {      # 详细损失组件
                'adversarial': [],
                'reconstruction': [],
                'constraint': [],
                'physics': []
            },
            'lr_history': {           # 学习率历史
                'generator': [],
                'discriminator': [],
                'forward_model': []
            }
        }
        
        print(f"Unified Trainer initialized on device: {self.device}")
    
    def initialize_models(self):
        """初始化所有模型"""
        print("Initializing models...")
        
        # 生成器
        self.generator = Generator(
            input_dim=cfg.SPECTRUM_DIM,
            output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM
        ).to(self.device)
        
        # 判别器
        self.discriminator = Discriminator(
            input_spec_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM,
            input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM
        ).to(self.device)
        
        # 前向模型
        self.forward_model = ForwardModel(
            input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
            output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
            output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM
        ).to(self.device)
        
        print("✓ All models initialized successfully")
        
        # 打印模型参数数量
        total_params = 0
        for name, model in [('Generator', self.generator), 
                           ('Discriminator', self.discriminator), 
                           ('Forward Model', self.forward_model)]:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params += params
            print(f"  - {name}: {params:,} parameters")
        print(f"  - Total: {total_params:,} parameters")
    
    def initialize_optimizers(self, mode: str = "full"):
        """
        初始化优化器和调度器
        
        Args:
            mode: "forward_only", "pigan_only", "full"
        """
        print(f"Initializing optimizers for {mode} training...")
        
        # 优化器配置
        gen_config = self.opt_config['optimizer']['generator']
        disc_config = self.opt_config['optimizer']['discriminator']
        fwd_config = self.opt_config['optimizer']['forward_model']
        
        if mode in ["forward_only", "full"]:
            self.optimizer_F = optim.Adam(
                self.forward_model.parameters(),
                lr=fwd_config['lr'],
                betas=fwd_config['betas'],
                weight_decay=fwd_config['weight_decay']
            )
            self.scheduler_F = CosineAnnealingLR(self.optimizer_F, T_max=100)
        
        if mode in ["pigan_only", "full"]:
            self.optimizer_G = optim.Adam(
                self.generator.parameters(),
                lr=gen_config['lr'],
                betas=gen_config['betas'],
                weight_decay=gen_config['weight_decay']
            )
            
            self.optimizer_D = optim.Adam(
                self.discriminator.parameters(),
                lr=disc_config['lr'],
                betas=disc_config['betas'],
                weight_decay=disc_config['weight_decay']
            )
            
            self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=200)
            self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=200)
        
        print("✓ Optimizers and schedulers initialized")
    
    def train_forward_model_only(self, dataloader, num_epochs: int = 100):
        """
        单独训练前向模型
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
        """
        print(f"\n=== Forward Model Training ({num_epochs} epochs) ===")
        
        self.initialize_optimizers("forward_only")
        self.forward_model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                real_spectrum, _, real_params_norm, _, real_metrics_norm = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                real_metrics_norm = real_metrics_norm.to(self.device)
                
                # 前向传播
                pred_spectrum, pred_metrics = self.forward_model(real_params_norm)
                
                # 损失计算
                spectrum_loss = self.criterion_mse(pred_spectrum, real_spectrum)
                metrics_loss = self.criterion_mse(pred_metrics, real_metrics_norm)
                
                # 平滑性损失
                spectrum_diff = torch.diff(pred_spectrum, dim=1)
                smoothness_loss = torch.mean(spectrum_diff ** 2)
                
                total_loss = (
                    self.opt_config['forward_model']['spectrum_loss_weight'] * spectrum_loss +
                    self.opt_config['forward_model']['metrics_loss_weight'] * metrics_loss +
                    self.opt_config['forward_model']['smoothness_loss_weight'] * smoothness_loss
                )
                
                # 反向传播
                self.optimizer_F.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
                self.optimizer_F.step()
                
                epoch_loss += total_loss.item()
            
            # 更新调度器
            self.scheduler_F.step()
            
            # 记录历史
            avg_loss = epoch_loss / len(dataloader)
            self.train_history['forward_losses'].append(avg_loss)
            self.train_history['lr_history']['forward_model'].append(self.scheduler_F.get_last_lr()[0])
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.6f}, LR: {self.scheduler_F.get_last_lr()[0]:.6f}")
        
        print("✓ Forward model training completed")
    
    def calculate_constraint_loss(self, pred_params_norm: torch.Tensor, param_ranges: Dict) -> torch.Tensor:
        """计算参数约束损失"""
        constraint_loss = 0.0
        
        # 硬约束：参数范围违约惩罚
        violation_penalty = torch.sum(
            torch.relu(pred_params_norm - 1.0) + torch.relu(-pred_params_norm)
        )
        
        # 软约束：边界平滑惩罚
        boundary_penalty = torch.sum(
            torch.exp(-10 * pred_params_norm) + torch.exp(-10 * (1 - pred_params_norm))
        )
        
        constraint_loss = (
            self.opt_config['constraints']['range_penalty_weight'] * violation_penalty +
            self.opt_config['constraints']['boundary_smoothness'] * boundary_penalty
        )
        
        return constraint_loss
    
    def calculate_physics_loss(self, pred_params_norm: torch.Tensor, real_spectrum: torch.Tensor) -> torch.Tensor:
        """计算物理约束损失"""
        physics_loss = 0.0
        
        # 前向一致性损失
        pred_spectrum, pred_metrics = self.forward_model(pred_params_norm)
        consistency_loss = self.criterion_mse(pred_spectrum, real_spectrum)
        
        # 物理合理性损失 - 谐振频率合理性
        freq_penalty = torch.sum(torch.relu(pred_metrics[:, 0] - 3.0) + torch.relu(0.5 - pred_metrics[:, 0]))
        
        physics_loss = (
            self.opt_config['loss_weights']['forward_consistency_loss'] * consistency_loss +
            self.opt_config['constraints']['physics_constraint_weight'] * freq_penalty
        )
        
        return physics_loss
    
    def calculate_stability_loss(self, pred_params_norm: torch.Tensor, real_spectrum: torch.Tensor) -> torch.Tensor:
        """计算稳定性损失"""
        # 添加小噪声测试稳定性
        noise = torch.randn_like(real_spectrum) * 0.01
        noisy_spectrum = real_spectrum + noise
        
        pred_params_noisy = self.generator(noisy_spectrum)
        stability_loss = self.criterion_mse(pred_params_norm, pred_params_noisy)
        
        return self.opt_config['loss_weights']['stability_loss'] * stability_loss
    
    def train_pigan_step(self, batch, dataset) -> Dict[str, float]:
        """PI-GAN训练单步"""
        real_spectrum, real_params_denorm, real_params_norm, real_metrics_denorm, real_metrics_norm = batch
        
        real_spectrum = real_spectrum.to(self.device)
        real_params_denorm = real_params_denorm.to(self.device)
        real_params_norm = real_params_norm.to(self.device)
        
        batch_size = real_spectrum.size(0)
        
        # =================== 训练判别器 ===================
        self.discriminator.train()
        self.optimizer_D.zero_grad()
        
        # 真实样本
        real_labels = torch.ones(batch_size, 1).to(self.device)
        real_scores = self.discriminator(real_spectrum, real_params_denorm)
        d_loss_real = self.criterion_bce(real_scores, real_labels)
        
        # 生成样本
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        with torch.no_grad():
            fake_params_norm = self.generator(real_spectrum)
        fake_params_denorm = denormalize_params(fake_params_norm, dataset.param_ranges)
        fake_scores = self.discriminator(real_spectrum, fake_params_denorm)
        d_loss_fake = self.criterion_bce(fake_scores, fake_labels)
        
        # 标签平滑
        label_smoothing = self.opt_config['discriminator']['label_smoothing']
        real_labels_smooth = torch.ones(batch_size, 1).to(self.device) * (1 - label_smoothing)
        fake_labels_smooth = torch.zeros(batch_size, 1).to(self.device) + label_smoothing
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer_D.step()
        
        # =================== 训练生成器 ===================
        self.generator.train()
        self.optimizer_G.zero_grad()
        
        # 生成参数
        pred_params_norm = self.generator(real_spectrum)
        pred_params_denorm = denormalize_params(pred_params_norm, dataset.param_ranges)
        
        # 对抗损失
        gen_scores = self.discriminator(real_spectrum, pred_params_denorm)
        g_loss_adv = self.criterion_bce(gen_scores, real_labels)
        
        # 重建损失
        g_loss_recon = self.criterion_mse(pred_params_norm, real_params_norm)
        
        # 约束损失
        g_loss_constraint = self.calculate_constraint_loss(pred_params_norm, dataset.param_ranges)
        
        # 物理损失
        g_loss_physics = self.calculate_physics_loss(pred_params_norm, real_spectrum)
        
        # 稳定性损失
        g_loss_stability = self.calculate_stability_loss(pred_params_norm, real_spectrum)
        
        # 总生成器损失
        g_loss = (
            self.opt_config['loss_weights']['adversarial_loss'] * g_loss_adv +
            self.opt_config['loss_weights']['reconstruction_loss'] * g_loss_recon +
            self.opt_config['loss_weights']['parameter_constraint_loss'] * g_loss_constraint +
            self.opt_config['loss_weights']['physics_constraint_loss'] * g_loss_physics +
            g_loss_stability
        )
        
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.optimizer_G.step()
        
        # 计算约束违约率
        with torch.no_grad():
            violations = torch.sum((pred_params_norm < 0) | (pred_params_norm > 1), dim=1)
            violation_rate = torch.mean((violations > 0).float()).item()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'g_loss_recon': g_loss_recon.item(),
            'g_loss_constraint': g_loss_constraint.item(),
            'g_loss_physics': g_loss_physics.item(),
            'g_loss_stability': g_loss_stability.item(),
            'violation_rate': violation_rate
        }
    
    def train_pigan_only(self, dataloader, dataset, num_epochs: int = 200):
        """
        单独训练PI-GAN
        
        Args:
            dataloader: 数据加载器
            dataset: 数据集
            num_epochs: 训练轮数
        """
        print(f"\n=== PI-GAN Training ({num_epochs} epochs) ===")
        
        self.initialize_optimizers("pigan_only")
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'g_loss': 0.0,
                'd_loss': 0.0,
                'violation_rate': 0.0,
                'g_loss_adv': 0.0,
                'g_loss_recon': 0.0,
                'g_loss_constraint': 0.0,
                'g_loss_physics': 0.0
            }
            
            for batch_idx, batch in enumerate(dataloader):
                losses = self.train_pigan_step(batch, dataset)
                
                for key in epoch_losses:
                    if key in losses:
                        epoch_losses[key] += losses[key]
            
            # 平均损失
            for key in epoch_losses:
                epoch_losses[key] /= len(dataloader)
            
            # 更新调度器
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # 记录历史
            self.train_history['g_losses'].append(epoch_losses['g_loss'])
            self.train_history['d_losses'].append(epoch_losses['d_loss'])
            self.train_history['constraint_violations'].append(epoch_losses['violation_rate'])
            self.train_history['detailed_losses']['adversarial'].append(epoch_losses['g_loss_adv'])
            self.train_history['detailed_losses']['reconstruction'].append(epoch_losses['g_loss_recon'])
            self.train_history['detailed_losses']['constraint'].append(epoch_losses['g_loss_constraint'])
            self.train_history['detailed_losses']['physics'].append(epoch_losses['g_loss_physics'])
            self.train_history['lr_history']['generator'].append(self.scheduler_G.get_last_lr()[0])
            self.train_history['lr_history']['discriminator'].append(self.scheduler_D.get_last_lr()[0])
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  G Loss: {epoch_losses['g_loss']:.6f} | D Loss: {epoch_losses['d_loss']:.6f}")
                print(f"  Violation Rate: {epoch_losses['violation_rate']:.4f}")
                print(f"  G LR: {self.scheduler_G.get_last_lr()[0]:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch + 1, mode="pigan")
        
        print("✓ PI-GAN training completed")
    
    def train_full_pipeline(self, dataloader, dataset, 
                           forward_epochs: int = 50, 
                           pigan_epochs: int = 200):
        """
        完整训练流水线：前向模型预训练 + PI-GAN训练
        
        Args:
            dataloader: 数据加载器
            dataset: 数据集
            forward_epochs: 前向模型预训练轮数
            pigan_epochs: PI-GAN训练轮数
        """
        print(f"\n{'='*60}")
        print("FULL TRAINING PIPELINE")
        print(f"{'='*60}")
        print(f"Phase 1: Forward Model Pre-training ({forward_epochs} epochs)")
        print(f"Phase 2: PI-GAN Training ({pigan_epochs} epochs)")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # 阶段1：前向模型预训练
        self.train_forward_model_only(dataloader, forward_epochs)
        
        # 阶段2：PI-GAN训练
        self.train_pigan_only(dataloader, dataset, pigan_epochs)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"FULL TRAINING COMPLETED in {total_time:.2f}s ({total_time/60:.1f}min)")
        print(f"{'='*60}")
        
        # 生成训练曲线
        self.plot_training_curves(mode="full")
    
    def plot_training_curves(self, mode: str = "full"):
        """生成训练曲线"""
        print("Generating training curves...")
        
        # 设置英文绘图
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        if mode == "forward_only":
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle('Forward Model Training Progress', fontsize=16)
            
            epochs = range(1, len(self.train_history['forward_losses']) + 1)
            
            # 前向模型损失
            axes[0].plot(epochs, self.train_history['forward_losses'], 'b-', linewidth=2)
            axes[0].set_title('Forward Model Loss')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].grid(True, alpha=0.3)
            
            # 学习率
            axes[1].plot(epochs, self.train_history['lr_history']['forward_model'], 'g-', linewidth=2)
            axes[1].set_title('Learning Rate Schedule')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Learning Rate')
            axes[1].set_yscale('log')
            axes[1].grid(True, alpha=0.3)
            
        elif mode == "pigan_only":
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('PI-GAN Training Progress', fontsize=16)
            
            epochs = range(1, len(self.train_history['g_losses']) + 1)
            
            # 生成器和判别器损失
            axes[0, 0].plot(epochs, self.train_history['g_losses'], 'b-', label='Generator', linewidth=2)
            axes[0, 0].plot(epochs, self.train_history['d_losses'], 'r-', label='Discriminator', linewidth=2)
            axes[0, 0].set_title('Generator vs Discriminator Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 约束违约率
            axes[0, 1].plot(epochs, self.train_history['constraint_violations'], 'orange', linewidth=2)
            axes[0, 1].axhline(y=0.1, color='green', linestyle='--', label='Target (10%)')
            axes[0, 1].set_title('Parameter Constraint Violation Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Violation Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # 损失组件分解
            colors = ['blue', 'green', 'orange', 'purple']
            for i, (component, color) in enumerate(zip(['adversarial', 'reconstruction', 'constraint', 'physics'], colors)):
                if self.train_history['detailed_losses'][component]:
                    axes[1, 0].plot(epochs, self.train_history['detailed_losses'][component], 
                                   color=color, label=f'{component.capitalize()}', linewidth=2)
            axes[1, 0].set_title('Loss Components Breakdown')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # 学习率
            axes[1, 1].plot(epochs, self.train_history['lr_history']['generator'], 'b-', label='Generator', linewidth=2)
            axes[1, 1].plot(epochs, self.train_history['lr_history']['discriminator'], 'r-', label='Discriminator', linewidth=2)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
        else:  # full mode
            fig, axes = plt.subplots(3, 2, figsize=(15, 15))
            fig.suptitle('Complete Training Pipeline Progress', fontsize=16)
            
            # 前向模型训练
            if self.train_history['forward_losses']:
                forward_epochs = range(1, len(self.train_history['forward_losses']) + 1)
                axes[0, 0].plot(forward_epochs, self.train_history['forward_losses'], 'b-', linewidth=2)
                axes[0, 0].set_title('Forward Model Training Loss')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
            
            # PI-GAN训练
            if self.train_history['g_losses']:
                pigan_epochs = range(1, len(self.train_history['g_losses']) + 1)
                
                # 生成器和判别器损失
                axes[0, 1].plot(pigan_epochs, self.train_history['g_losses'], 'b-', label='Generator', linewidth=2)
                axes[0, 1].plot(pigan_epochs, self.train_history['d_losses'], 'r-', label='Discriminator', linewidth=2)
                axes[0, 1].set_title('PI-GAN Training: Generator vs Discriminator')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # 约束违约率
                axes[1, 0].plot(pigan_epochs, self.train_history['constraint_violations'], 'orange', linewidth=2)
                axes[1, 0].axhline(y=0.1, color='green', linestyle='--', label='Target (10%)')
                axes[1, 0].set_title('Parameter Constraint Violation Rate')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Violation Rate')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # 损失组件分解
                colors = ['blue', 'green', 'orange', 'purple']
                for component, color in zip(['adversarial', 'reconstruction', 'constraint', 'physics'], colors):
                    if self.train_history['detailed_losses'][component]:
                        axes[1, 1].plot(pigan_epochs, self.train_history['detailed_losses'][component], 
                                       color=color, label=f'{component.capitalize()}', linewidth=2)
                axes[1, 1].set_title('Loss Components Breakdown')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss Value')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                # 学习率 - 前向模型
                if self.train_history['lr_history']['forward_model']:
                    axes[2, 0].plot(forward_epochs, self.train_history['lr_history']['forward_model'], 'g-', linewidth=2)
                    axes[2, 0].set_title('Forward Model Learning Rate')
                    axes[2, 0].set_xlabel('Epoch')
                    axes[2, 0].set_ylabel('Learning Rate')
                    axes[2, 0].set_yscale('log')
                    axes[2, 0].grid(True, alpha=0.3)
                
                # 学习率 - PI-GAN
                axes[2, 1].plot(pigan_epochs, self.train_history['lr_history']['generator'], 'b-', label='Generator', linewidth=2)
                axes[2, 1].plot(pigan_epochs, self.train_history['lr_history']['discriminator'], 'r-', label='Discriminator', linewidth=2)
                axes[2, 1].set_title('PI-GAN Learning Rates')
                axes[2, 1].set_xlabel('Epoch')
                axes[2, 1].set_ylabel('Learning Rate')
                axes[2, 1].set_yscale('log')
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f"unified_training_curves_{mode}_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to: {plot_path}")
    
    def save_checkpoint(self, epoch: int, mode: str = "full"):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'mode': mode,
            'train_history': self.train_history,
            'optimization_config': self.opt_config
        }
        
        if self.generator is not None:
            checkpoint['generator_state_dict'] = self.generator.state_dict()
        if self.discriminator is not None:
            checkpoint['discriminator_state_dict'] = self.discriminator.state_dict()
        if self.forward_model is not None:
            checkpoint['forward_model_state_dict'] = self.forward_model.state_dict()
        
        if self.optimizer_G is not None:
            checkpoint['optimizer_G_state_dict'] = self.optimizer_G.state_dict()
        if self.optimizer_D is not None:
            checkpoint['optimizer_D_state_dict'] = self.optimizer_D.state_dict()
        if self.optimizer_F is not None:
            checkpoint['optimizer_F_state_dict'] = self.optimizer_F.state_dict()
        
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f'unified_checkpoint_{mode}_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_models(self):
        """保存最终模型"""
        os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
        
        # 保存为评估器期望的文件名
        if self.generator is not None:
            torch.save(self.generator.state_dict(), 
                      os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth"))
        if self.discriminator is not None:
            torch.save(self.discriminator.state_dict(), 
                      os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth"))
        if self.forward_model is not None:
            torch.save(self.forward_model.state_dict(), 
                      os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth"))
        
        # 同时保存统一版本作为备份
        if self.generator is not None:
            torch.save(self.generator.state_dict(), 
                      os.path.join(cfg.SAVED_MODELS_DIR, "generator_unified.pth"))
        if self.discriminator is not None:
            torch.save(self.discriminator.state_dict(), 
                      os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_unified.pth"))
        if self.forward_model is not None:
            torch.save(self.forward_model.state_dict(), 
                      os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_unified.pth"))
        
        print("✓ Final models saved")
        print(f"✓ Models saved to: {cfg.SAVED_MODELS_DIR}")
        print("  - generator_final.pth")
        print("  - discriminator_final.pth") 
        print("  - forward_model_final.pth")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'generator_state_dict' in checkpoint and self.generator is not None:
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
        if 'discriminator_state_dict' in checkpoint and self.discriminator is not None:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        if 'forward_model_state_dict' in checkpoint and self.forward_model is not None:
            self.forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
        
        if 'train_history' in checkpoint:
            self.train_history = checkpoint['train_history']
        
        print("✓ Checkpoint loaded successfully")
        return checkpoint.get('epoch', 0)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Unified PI-GAN Training System")
    parser.add_argument('--mode', type=str, default='full', 
                        choices=['forward_only', 'pigan_only', 'full'],
                        help='Training mode: forward_only, pigan_only, or full')
    parser.add_argument('--forward_epochs', type=int, default=50, 
                        help='Number of forward model training epochs')
    parser.add_argument('--pigan_epochs', type=int, default=200, 
                        help='Number of PI-GAN training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建数据加载器
    dataset = MetamaterialDataset(cfg.DATASET_PATH, cfg.SPECTRUM_DIM)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    print(f"Batch size: {args.batch_size}, Batches per epoch: {len(dataloader)}")
    
    # 创建统一训练器
    trainer = UnifiedTrainer(device=args.device)
    trainer.initialize_models()
    
    # 恢复检查点（如果指定）
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # 根据模式进行训练
    if args.mode == 'forward_only':
        trainer.train_forward_model_only(dataloader, args.forward_epochs)
    elif args.mode == 'pigan_only':
        trainer.train_pigan_only(dataloader, dataset, args.pigan_epochs)
    else:  # full mode
        trainer.train_full_pipeline(dataloader, dataset, args.forward_epochs, args.pigan_epochs)
    
    # 保存最终模型
    trainer.save_final_models()
    
    print(f"\n🎉 {args.mode.upper()} training completed successfully!")

if __name__ == "__main__":
    main()
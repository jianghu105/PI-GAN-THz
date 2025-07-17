# PI_GAN_THZ/core/train/optimized_trainer.py
# 优化训练器 - 基于评估结果的改进版本

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
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

class OptimizedTrainer:
    """
    优化训练器 - 针对评估结果的改进版本
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
            'g_losses': [],
            'd_losses': [],
            'f_losses': [],
            'constraint_violations': [],
            'evaluation_metrics': []
        }
        
        print(f"Optimized Trainer initialized on device: {self.device}")
    
    def initialize_models(self):
        """初始化优化后的模型"""
        print("Initializing optimized models...")
        
        # 生成器 - 基础架构（原模型不支持额外参数）
        self.generator = Generator(
            input_dim=cfg.SPECTRUM_DIM,
            output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM
        ).to(self.device)
        
        # 判别器 - 基础架构（原模型不支持额外参数）
        self.discriminator = Discriminator(
            input_spec_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM,
            input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM
        ).to(self.device)
        
        # 前向模型 - 基础架构（原模型不支持额外参数）
        self.forward_model = ForwardModel(
            input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
            output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
            output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM
        ).to(self.device)
        
        print("✓ Models initialized with optimized architectures")
    
    def initialize_optimizers(self):
        """初始化优化器和调度器"""
        print("Initializing optimizers and schedulers...")
        
        # 优化器
        gen_config = self.opt_config['optimizer']['generator']
        disc_config = self.opt_config['optimizer']['discriminator']
        fwd_config = self.opt_config['optimizer']['forward_model']
        
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
        
        self.optimizer_F = optim.Adam(
            self.forward_model.parameters(),
            lr=fwd_config['lr'],
            betas=fwd_config['betas'],
            weight_decay=fwd_config['weight_decay']
        )
        
        # 学习率调度器
        training_config = self.opt_config['training']
        total_epochs = training_config.get('total_epochs', 200)
        
        self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=total_epochs)
        self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=total_epochs)
        self.scheduler_F = CosineAnnealingLR(self.optimizer_F, T_max=total_epochs)
        
        print("✓ Optimizers and schedulers initialized")
    
    def calculate_constraint_loss(self, pred_params_norm: torch.Tensor, 
                                 param_ranges: Dict) -> torch.Tensor:
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
    
    def calculate_physics_loss(self, pred_params_norm: torch.Tensor, 
                              real_spectrum: torch.Tensor) -> torch.Tensor:
        """计算物理约束损失"""
        physics_loss = 0.0
        
        # 前向一致性损失
        pred_spectrum, pred_metrics = self.forward_model(pred_params_norm)
        consistency_loss = self.criterion_mse(pred_spectrum, real_spectrum)
        
        # 物理合理性损失
        # 检查谐振频率的合理性
        freq_penalty = torch.sum(torch.relu(pred_metrics[:, 0] - 3.0) + torch.relu(0.5 - pred_metrics[:, 0]))
        
        physics_loss = (
            self.opt_config['loss_weights']['forward_consistency_loss'] * consistency_loss +
            self.opt_config['constraints']['physics_constraint_weight'] * freq_penalty
        )
        
        return physics_loss
    
    def calculate_stability_loss(self, pred_params_norm: torch.Tensor,
                                real_spectrum: torch.Tensor) -> torch.Tensor:
        """计算稳定性损失"""
        # 添加小噪声测试稳定性
        noise = torch.randn_like(real_spectrum) * 0.01
        noisy_spectrum = real_spectrum + noise
        
        pred_params_noisy = self.generator(noisy_spectrum)
        stability_loss = self.criterion_mse(pred_params_norm, pred_params_noisy)
        
        return self.opt_config['loss_weights']['stability_loss'] * stability_loss
    
    def train_forward_model(self, dataloader, num_epochs: int = 100):
        """预训练前向模型"""
        print(f"\nPre-training Forward Model for {num_epochs} epochs...")
        
        self.forward_model.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
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
                
                total_loss_batch = (
                    self.opt_config['forward_model']['spectrum_loss_weight'] * spectrum_loss +
                    self.opt_config['forward_model']['metrics_loss_weight'] * metrics_loss +
                    self.opt_config['forward_model']['smoothness_loss_weight'] * smoothness_loss
                )
                
                # 反向传播
                self.optimizer_F.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
                self.optimizer_F.step()
                
                total_loss += total_loss_batch.item()
            
            # 调度器更新
            self.scheduler_F.step()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch [{epoch+1}/{num_epochs}], Forward Model Loss: {avg_loss:.6f}")
        
        print("✓ Forward Model pre-training completed")
    
    def train_step(self, batch, dataset) -> Dict[str, float]:
        """单步训练"""
        real_spectrum, real_params_denorm, real_params_norm, real_metrics_denorm, real_metrics_norm = batch
        
        real_spectrum = real_spectrum.to(self.device)
        real_params_denorm = real_params_denorm.to(self.device)
        real_params_norm = real_params_norm.to(self.device)
        real_metrics_norm = real_metrics_norm.to(self.device)
        
        batch_size = real_spectrum.size(0)
        
        # ===================
        # 训练判别器
        # ===================
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
        
        # ===================
        # 训练生成器
        # ===================
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
            'violation_rate': violation_rate
        }
    
    def train(self, dataloader, dataset, num_epochs: int = 200):
        """主训练循环"""
        print(f"\nStarting optimized training for {num_epochs} epochs...")
        
        # 预训练前向模型
        self.train_forward_model(dataloader, num_epochs=50)
        
        # 主训练循环
        for epoch in range(num_epochs):
            epoch_losses = {
                'g_loss': 0.0,
                'd_loss': 0.0,
                'violation_rate': 0.0
            }
            
            for batch_idx, batch in enumerate(dataloader):
                losses = self.train_step(batch, dataset)
                
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
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  G Loss: {epoch_losses['g_loss']:.6f}")
                print(f"  D Loss: {epoch_losses['d_loss']:.6f}")
                print(f"  Violation Rate: {epoch_losses['violation_rate']:.4f}")
                print(f"  G LR: {self.scheduler_G.get_last_lr()[0]:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch + 1)
        
        print("✓ Optimized training completed")
        
        # Generate training plots
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Generate training loss and accuracy curves"""
        if not self.train_history['g_losses']:
            print("⚠ No training history available for plotting")
            return
        
        # Set up English plotting
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PI-GAN Training Progress', fontsize=16)
        
        epochs = range(1, len(self.train_history['g_losses']) + 1)
        
        # 1. Generator and Discriminator Loss
        axes[0, 0].plot(epochs, self.train_history['g_losses'], label='Generator Loss', color='blue', alpha=0.8)
        axes[0, 0].plot(epochs, self.train_history['d_losses'], label='Discriminator Loss', color='red', alpha=0.8)
        axes[0, 0].set_title('Generator vs Discriminator Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Parameter Violation Rate
        if self.train_history['constraint_violations']:
            axes[0, 1].plot(epochs, self.train_history['constraint_violations'], 
                           label='Violation Rate', color='orange', linewidth=2)
            axes[0, 1].axhline(y=0.1, color='green', linestyle='--', 
                              label='Target (10%)', alpha=0.7)
            axes[0, 1].set_title('Parameter Constraint Violation Rate')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Violation Rate')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Training Loss Breakdown (if available)
        if hasattr(self, 'detailed_losses') and self.detailed_losses:
            loss_components = ['adversarial', 'reconstruction', 'constraint', 'physics']
            colors = ['blue', 'green', 'orange', 'purple']
            for i, (component, color) in enumerate(zip(loss_components, colors)):
                if component in self.detailed_losses:
                    axes[1, 0].plot(epochs, self.detailed_losses[component], 
                                   label=f'{component.capitalize()} Loss', color=color, alpha=0.8)
            axes[1, 0].set_title('Loss Components Breakdown')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Detailed loss breakdown\nnot available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes,
                           fontsize=12, alpha=0.7)
            axes[1, 0].set_title('Loss Components Breakdown')
        
        # 4. Learning Rate Schedule
        if hasattr(self, 'lr_history') and self.lr_history:
            axes[1, 1].plot(epochs, self.lr_history['generator'], 
                           label='Generator LR', color='blue', alpha=0.8)
            axes[1, 1].plot(epochs, self.lr_history['discriminator'], 
                           label='Discriminator LR', color='red', alpha=0.8)
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_yscale('log')
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning rate history\nnot available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes,
                           fontsize=12, alpha=0.7)
            axes[1, 1].set_title('Learning Rate Schedule')
        
        plt.tight_layout()
        
        # Save plot
        plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f"training_curves_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training curves saved to: {plot_path}")
    
    def save_checkpoint(self, epoch: int):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'forward_model_state_dict': self.forward_model.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_F_state_dict': self.optimizer_F.state_dict(),
            'train_history': self.train_history,
            'optimization_config': self.opt_config
        }
        
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f'optimized_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def save_final_models(self):
        """保存最终模型"""
        os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
        
        torch.save(self.generator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "generator_optimized.pth"))
        torch.save(self.discriminator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_optimized.pth"))
        torch.save(self.forward_model.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_optimized.pth"))
        
        print("✓ Final optimized models saved")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Optimized PI-GAN Training")
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建数据加载器
    dataset = MetamaterialDataset(cfg.DATASET_PATH, cfg.SPECTRUM_DIM)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=2,  # Reduced to avoid worker warning
        pin_memory=True
    )
    
    # 创建优化训练器
    trainer = OptimizedTrainer(device=args.device)
    trainer.initialize_models()
    trainer.initialize_optimizers()
    
    # 开始训练
    trainer.train(dataloader, dataset, num_epochs=args.epochs)
    
    # 保存最终模型
    trainer.save_final_models()
    
    print("🎉 Optimized training completed successfully!")

if __name__ == "__main__":
    main()
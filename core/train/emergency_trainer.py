# PI_GAN_THZ/core/train/emergency_trainer.py
# 紧急修复训练器 - 针对当前评估结果的问题

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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

class EmergencyTrainer:
    """
    紧急修复训练器 - 针对当前问题的专门解决方案
    
    针对问题:
    1. 前向网络R²=-0.1768 (极差)
    2. 生成器R²=-0.3637 (极差) 
    3. 循环一致性=0.2062 (严重)
    4. 判别器过强=0.9225
    """
    
    def __init__(self, device: str = "auto"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
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
        self.criterion_l1 = nn.L1Loss()
        
        # 紧急修复配置
        self.emergency_config = {
            # 前向网络强化训练
            'forward_intensive_epochs': 200,  # 大幅增加前向网络训练
            'forward_lr': 5e-4,              # 提高前向网络学习率
            
            # 平衡判别器和生成器
            'discriminator_lr': 5e-5,         # 大幅降低判别器学习率
            'generator_lr': 2e-4,             # 保持生成器学习率
            'discriminator_update_freq': 2,   # 判别器每2轮更新一次
            
            # 损失权重重新平衡
            'forward_consistency_weight': 20.0,  # 大幅提高一致性权重
            'reconstruction_weight': 15.0,       # 提高重建权重
            'adversarial_weight': 0.1,          # 大幅降低对抗权重
            'l1_penalty_weight': 5.0,           # 添加L1正则化
            
            # 渐进式训练
            'warmup_epochs': 100,               # 100轮warmup
            'progressive_adversarial': True,    # 渐进式对抗训练
        }
        
        # 训练历史
        self.train_history = {
            'forward_losses': [],
            'g_losses': [],
            'd_losses': [],
            'consistency_errors': [],
            'r2_scores': [],
            'lr_history': {'generator': [], 'discriminator': [], 'forward_model': []}
        }
        
        print(f"Emergency Trainer initialized on device: {self.device}")
        print("🚨 针对当前问题的紧急修复模式")
        
    def initialize_models(self):
        """初始化模型"""
        print("Initializing models for emergency training...")
        
        self.generator = Generator(
            input_dim=cfg.SPECTRUM_DIM,
            output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM
        ).to(self.device)
        
        self.discriminator = Discriminator(
            input_spec_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM,
            input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM
        ).to(self.device)
        
        self.forward_model = ForwardModel(
            input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
            output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
            output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM
        ).to(self.device)
        
        print("✓ Models initialized")
        
    def initialize_optimizers(self):
        """初始化优化器 - 使用紧急修复配置"""
        print("Initializing emergency optimizers...")
        
        # 前向模型 - 提高学习率
        self.optimizer_F = optim.Adam(
            self.forward_model.parameters(),
            lr=self.emergency_config['forward_lr'],
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        self.scheduler_F = ReduceLROnPlateau(
            self.optimizer_F, mode='min', factor=0.5, patience=20, verbose=True
        )
        
        # 生成器 - 保持学习率
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.emergency_config['generator_lr'],
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        self.scheduler_G = ReduceLROnPlateau(
            self.optimizer_G, mode='min', factor=0.7, patience=15, verbose=True
        )
        
        # 判别器 - 大幅降低学习率
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.emergency_config['discriminator_lr'],
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        self.scheduler_D = ReduceLROnPlateau(
            self.optimizer_D, mode='min', factor=0.8, patience=25, verbose=True
        )
        
        print("✓ Emergency optimizers initialized")
        print(f"  - Forward Model LR: {self.emergency_config['forward_lr']}")
        print(f"  - Generator LR: {self.emergency_config['generator_lr']}")
        print(f"  - Discriminator LR: {self.emergency_config['discriminator_lr']} (大幅降低)")
        
    def intensive_forward_training(self, dataloader, num_epochs: int = 200):
        """
        前向网络强化训练 - 解决R²=-0.1768问题
        """
        print(f"\n🔥 INTENSIVE FORWARD MODEL TRAINING ({num_epochs} epochs)")
        print("目标: 解决前向网络R²=-0.1768的严重问题")
        
        self.forward_model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            spectrum_loss_sum = 0.0
            metrics_loss_sum = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                real_spectrum, _, real_params_norm, _, real_metrics_norm = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                real_metrics_norm = real_metrics_norm.to(self.device)
                
                # 前向传播
                pred_spectrum, pred_metrics = self.forward_model(real_params_norm)
                
                # 多重损失
                spectrum_loss = self.criterion_mse(pred_spectrum, real_spectrum)
                metrics_loss = self.criterion_mse(pred_metrics, real_metrics_norm)
                
                # L1正则化增强泛化能力
                spectrum_l1_loss = self.criterion_l1(pred_spectrum, real_spectrum)
                metrics_l1_loss = self.criterion_l1(pred_metrics, real_metrics_norm)
                
                # 平滑性损失
                spectrum_diff = torch.diff(pred_spectrum, dim=1)
                smoothness_loss = torch.mean(spectrum_diff ** 2)
                
                # 总损失 - 重新平衡权重
                total_loss = (
                    5.0 * spectrum_loss +        # 提高光谱权重
                    3.0 * metrics_loss +         # 提高指标权重
                    2.0 * spectrum_l1_loss +     # 添加L1正则
                    1.0 * metrics_l1_loss +      # 添加L1正则
                    0.5 * smoothness_loss        # 平滑性
                )
                
                # 反向传播
                self.optimizer_F.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 0.5)
                self.optimizer_F.step()
                
                epoch_loss += total_loss.item()
                spectrum_loss_sum += spectrum_loss.item()
                metrics_loss_sum += metrics_loss.item()
            
            # 平均损失
            avg_loss = epoch_loss / len(dataloader)
            avg_spectrum_loss = spectrum_loss_sum / len(dataloader)
            avg_metrics_loss = metrics_loss_sum / len(dataloader)
            
            # 记录历史
            self.train_history['forward_losses'].append(avg_loss)
            self.train_history['lr_history']['forward_model'].append(
                self.optimizer_F.param_groups[0]['lr']
            )
            
            # 学习率调度
            self.scheduler_F.step(avg_loss)
            
            # 早停检查
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # 保存最佳前向模型
                torch.save(self.forward_model.state_dict(), 
                          os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_best.pth"))
            else:
                patience_counter += 1
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Total Loss: {avg_loss:.6f}")
                print(f"  Spectrum Loss: {avg_spectrum_loss:.6f}")
                print(f"  Metrics Loss: {avg_metrics_loss:.6f}")
                print(f"  LR: {self.optimizer_F.param_groups[0]['lr']:.6f}")
                print(f"  Best Loss: {best_loss:.6f}")
            
            # 早停
            if patience_counter >= 30:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print("✓ Intensive forward model training completed")
        print(f"  Final Loss: {avg_loss:.6f}")
        print(f"  Best Loss: {best_loss:.6f}")
        
    def balanced_gan_training(self, dataloader, dataset, num_epochs: int = 200):
        """
        平衡GAN训练 - 解决判别器过强问题
        """
        print(f"\n⚖️ BALANCED GAN TRAINING ({num_epochs} epochs)")
        print("目标: 平衡判别器(当前92.25%)和生成器性能")
        
        discriminator_update_counter = 0
        
        for epoch in range(num_epochs):
            epoch_losses = {
                'g_loss': 0.0,
                'd_loss': 0.0,
                'consistency_error': 0.0,
                'r2_score': 0.0
            }
            
            for batch_idx, batch in enumerate(dataloader):
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_denorm = real_params_denorm.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                batch_size = real_spectrum.size(0)
                
                # ===================
                # 训练生成器 (每轮都训练)
                # ===================
                self.generator.train()
                self.optimizer_G.zero_grad()
                
                # 生成参数
                pred_params_norm = self.generator(real_spectrum)
                pred_params_denorm = denormalize_params(pred_params_norm, dataset.param_ranges)
                
                # 对抗损失 (大幅降低权重)
                if epoch >= self.emergency_config['warmup_epochs']:
                    gen_scores = self.discriminator(real_spectrum, pred_params_denorm)
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    g_loss_adv = self.criterion_bce(gen_scores, real_labels)
                    adversarial_weight = self.emergency_config['adversarial_weight']
                else:
                    g_loss_adv = torch.tensor(0.0).to(self.device)
                    adversarial_weight = 0.0  # warmup期间不使用对抗损失
                
                # 重建损失 (提高权重)
                g_loss_recon = self.criterion_mse(pred_params_norm, real_params_norm)
                g_loss_recon_l1 = self.criterion_l1(pred_params_norm, real_params_norm)
                
                # 前向一致性损失 (大幅提高权重)
                pred_spectrum_from_params, _ = self.forward_model(pred_params_norm)
                g_loss_consistency = self.criterion_mse(pred_spectrum_from_params, real_spectrum)
                
                # 总生成器损失
                g_loss = (
                    adversarial_weight * g_loss_adv +
                    self.emergency_config['reconstruction_weight'] * g_loss_recon +
                    self.emergency_config['l1_penalty_weight'] * g_loss_recon_l1 +
                    self.emergency_config['forward_consistency_weight'] * g_loss_consistency
                )
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 0.5)
                self.optimizer_G.step()
                
                # 计算R²分数
                with torch.no_grad():
                    real_flat = real_params_norm.cpu().numpy().flatten()
                    pred_flat = pred_params_norm.detach().cpu().numpy().flatten()
                    
                    # 简单R²计算
                    ss_res = np.sum((real_flat - pred_flat) ** 2)
                    ss_tot = np.sum((real_flat - np.mean(real_flat)) ** 2)
                    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
                
                # ===================
                # 训练判别器 (降低频率)
                # ===================
                discriminator_update_counter += 1
                if discriminator_update_counter % self.emergency_config['discriminator_update_freq'] == 0:
                    self.discriminator.train()
                    self.optimizer_D.zero_grad()
                    
                    # 真实样本
                    real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9  # 标签平滑
                    real_scores = self.discriminator(real_spectrum, real_params_denorm)
                    d_loss_real = self.criterion_bce(real_scores, real_labels)
                    
                    # 生成样本
                    fake_labels = torch.zeros(batch_size, 1).to(self.device) + 0.1  # 标签平滑
                    with torch.no_grad():
                        fake_params_norm = self.generator(real_spectrum)
                    fake_params_denorm = denormalize_params(fake_params_norm, dataset.param_ranges)
                    fake_scores = self.discriminator(real_spectrum, fake_params_denorm)
                    d_loss_fake = self.criterion_bce(fake_scores, fake_labels)
                    
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                    self.optimizer_D.step()
                else:
                    d_loss = torch.tensor(0.0)
                
                # 累积损失
                epoch_losses['g_loss'] += g_loss.item()
                epoch_losses['d_loss'] += d_loss.item()
                epoch_losses['consistency_error'] += g_loss_consistency.item()
                epoch_losses['r2_score'] += r2_score
            
            # 平均损失
            for key in epoch_losses:
                epoch_losses[key] /= len(dataloader)
            
            # 记录历史
            self.train_history['g_losses'].append(epoch_losses['g_loss'])
            self.train_history['d_losses'].append(epoch_losses['d_loss'])
            self.train_history['consistency_errors'].append(epoch_losses['consistency_error'])
            self.train_history['r2_scores'].append(epoch_losses['r2_score'])
            self.train_history['lr_history']['generator'].append(
                self.optimizer_G.param_groups[0]['lr']
            )
            self.train_history['lr_history']['discriminator'].append(
                self.optimizer_D.param_groups[0]['lr']
            )
            
            # 学习率调度
            self.scheduler_G.step(epoch_losses['g_loss'])
            self.scheduler_D.step(epoch_losses['d_loss'])
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  G Loss: {epoch_losses['g_loss']:.6f}")
                print(f"  D Loss: {epoch_losses['d_loss']:.6f}")
                print(f"  Consistency Error: {epoch_losses['consistency_error']:.6f}")
                print(f"  R² Score: {epoch_losses['r2_score']:.4f}")
                print(f"  G LR: {self.optimizer_G.param_groups[0]['lr']:.6f}")
                
                # 判断训练状态
                if epoch_losses['r2_score'] > 0.1:
                    print("  🟢 R²分数开始改善")
                elif epoch_losses['r2_score'] > -0.1:
                    print("  🟡 R²分数接近正值")
                else:
                    print("  🔴 R²分数仍为负值")
            
            # 保存检查点
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch + 1, "emergency")
        
        print("✓ Balanced GAN training completed")
        
    def emergency_full_training(self, dataloader, dataset):
        """
        完整紧急修复训练流程
        """
        print(f"\n{'='*80}")
        print("🚨 EMERGENCY TRAINING PIPELINE")
        print(f"{'='*80}")
        print("阶段1: 前向网络强化训练 (200轮)")
        print("阶段2: 平衡GAN训练 (200轮)")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        # 阶段1: 前向网络强化训练
        self.intensive_forward_training(dataloader, self.emergency_config['forward_intensive_epochs'])
        
        # 阶段2: 平衡GAN训练
        self.balanced_gan_training(dataloader, dataset, 200)
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"🚨 EMERGENCY TRAINING COMPLETED in {total_time:.2f}s ({total_time/60:.1f}min)")
        print(f"{'='*80}")
        
        # 生成训练曲线
        self.plot_emergency_curves()
        
    def plot_emergency_curves(self):
        """生成紧急训练曲线"""
        print("Generating emergency training curves...")
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        fig.suptitle('Emergency Training Progress - Problem Solving', fontsize=16)
        
        # 1. 前向模型损失
        if self.train_history['forward_losses']:
            forward_epochs = range(1, len(self.train_history['forward_losses']) + 1)
            axes[0, 0].plot(forward_epochs, self.train_history['forward_losses'], 'b-', linewidth=2)
            axes[0, 0].set_title('Forward Model Intensive Training')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_yscale('log')
        
        # 2. GAN损失
        if self.train_history['g_losses']:
            gan_epochs = range(1, len(self.train_history['g_losses']) + 1)
            axes[0, 1].plot(gan_epochs, self.train_history['g_losses'], 'b-', label='Generator', linewidth=2)
            axes[0, 1].plot(gan_epochs, self.train_history['d_losses'], 'r-', label='Discriminator', linewidth=2)
            axes[0, 1].set_title('Balanced GAN Training')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 一致性误差改善
        if self.train_history['consistency_errors']:
            axes[1, 0].plot(gan_epochs, self.train_history['consistency_errors'], 'orange', linewidth=2)
            axes[1, 0].axhline(y=0.01, color='green', linestyle='--', label='Target (<0.01)')
            axes[1, 0].set_title('Cycle Consistency Error Improvement')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Consistency Error')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_yscale('log')
        
        # 4. R²分数改善
        if self.train_history['r2_scores']:
            axes[1, 1].plot(gan_epochs, self.train_history['r2_scores'], 'purple', linewidth=2)
            axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5, label='Zero Line')
            axes[1, 1].axhline(y=0.8, color='green', linestyle='--', label='Target (>0.8)')
            axes[1, 1].set_title('Parameter Prediction R² Recovery')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 5. 学习率调度
        if self.train_history['lr_history']['forward_model']:
            axes[2, 0].plot(forward_epochs, self.train_history['lr_history']['forward_model'], 
                           'g-', label='Forward Model', linewidth=2)
            if self.train_history['lr_history']['generator']:
                axes[2, 0].plot(gan_epochs, self.train_history['lr_history']['generator'], 
                               'b-', label='Generator', linewidth=2)
                axes[2, 0].plot(gan_epochs, self.train_history['lr_history']['discriminator'], 
                               'r-', label='Discriminator (Reduced)', linewidth=2)
            axes[2, 0].set_title('Learning Rate Schedules')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Learning Rate')
            axes[2, 0].set_yscale('log')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        
        # 6. 问题解决总结
        axes[2, 1].axis('off')
        
        summary_text = """
Emergency Training Summary:

🔴 Identified Problems:
• Forward Network R²: -0.1768 (Collapsed)
• Generator R²: -0.3637 (Collapsed)  
• Cycle Consistency: 0.2062 (Very Poor)
• Discriminator: 0.9225 (Too Strong)

🟢 Solutions Applied:
• Intensive forward training (200 epochs)
• Reduced discriminator learning rate (5x)
• Increased consistency loss weight (20x)
• Added L1 regularization
• Progressive adversarial training
• Balanced update frequencies

🎯 Expected Improvements:
• Forward R²: -0.18 → >0.80
• Generator R²: -0.36 → >0.70  
• Consistency: 0.21 → <0.05
• Balanced discriminator accuracy
"""
        
        axes[2, 1].text(0.05, 0.95, summary_text, transform=axes[2, 1].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存图片
        plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f"emergency_training_curves_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Emergency training curves saved to: {plot_path}")
    
    def save_checkpoint(self, epoch: int, mode: str = "emergency"):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'mode': mode,
            'train_history': self.train_history,
            'emergency_config': self.emergency_config,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'forward_model_state_dict': self.forward_model.state_dict(),
        }
        
        if self.optimizer_G is not None:
            checkpoint['optimizer_G_state_dict'] = self.optimizer_G.state_dict()
        if self.optimizer_D is not None:
            checkpoint['optimizer_D_state_dict'] = self.optimizer_D.state_dict()
        if self.optimizer_F is not None:
            checkpoint['optimizer_F_state_dict'] = self.optimizer_F.state_dict()
        
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f'emergency_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Emergency checkpoint saved: {checkpoint_path}")
    
    def save_final_models(self):
        """保存最终模型"""
        os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
        
        # 保存为评估器期望的文件名
        torch.save(self.generator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth"))
        torch.save(self.discriminator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth"))
        torch.save(self.forward_model.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth"))
        
        # 同时保存紧急修复版本
        torch.save(self.generator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "generator_emergency.pth"))
        torch.save(self.discriminator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_emergency.pth"))
        torch.save(self.forward_model.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_emergency.pth"))
        
        print("✓ Emergency training models saved")
        print(f"✓ Models saved to: {cfg.SAVED_MODELS_DIR}")
        print("  - *_final.pth (for evaluator)")
        print("  - *_emergency.pth (backup)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Emergency PI-GAN Training - Problem Solver")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (reduced for stability)')
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
        num_workers=2,
        pin_memory=True
    )
    
    print(f"Emergency training dataset: {len(dataset)} samples")
    print(f"Batch size: {args.batch_size} (reduced for stability)")
    
    # 创建紧急训练器
    trainer = EmergencyTrainer(device=args.device)
    trainer.initialize_models()
    trainer.initialize_optimizers()
    
    # 开始紧急修复训练
    trainer.emergency_full_training(dataloader, dataset)
    
    # 保存最终模型
    trainer.save_final_models()
    
    print(f"\n🚨 Emergency training completed!")
    print("🔍 建议立即运行评估验证改进效果:")
    print("python core/evaluate/unified_evaluator.py --num_samples 1000")

if __name__ == "__main__":
    main()
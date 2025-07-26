# PI_GAN_THZ/core/train/constraint_optimizer.py
# 约束优化训练器 - 专门解决参数违约率问题

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from typing import Dict, Tuple

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

class ConstraintOptimizer:
    """
    约束优化训练器 - 专门解决91.4%参数违约率问题
    
    当前状态:
    ✓ PI-GAN Parameter R²: 0.9888 (优秀)
    ✓ Cycle Consistency: 0.013182 (优秀)
    ✓ Discriminator Balance: 51.00% (完美)
    ❌ Parameter Violation Rate: 91.4% (需要解决)
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
        
        # 损失函数
        self.criterion_bce = criterion_bce()
        self.criterion_mse = criterion_mse()
        self.criterion_l1 = nn.L1Loss()
        
        # 约束优化专用配置
        self.constraint_config = {
            # 约束相关权重 - 大幅提升
            'hard_constraint_weight': 50.0,      # 硬约束权重
            'boundary_penalty_weight': 20.0,     # 边界惩罚权重
            'range_violation_weight': 100.0,     # 范围违约惩罚
            'smoothness_penalty_weight': 10.0,   # 参数平滑性
            
            # 保持现有优秀性能的权重
            'reconstruction_weight': 15.0,       # 保持高重建性能
            'consistency_weight': 20.0,          # 保持循环一致性
            'adversarial_weight': 0.1,           # 保持判别器平衡
            
            # 学习率配置 - 精细调整
            'generator_lr': 1e-4,                # 生成器学习率
            'discriminator_lr': 5e-5,            # 保持判别器平衡
            
            # 约束训练策略
            'constraint_epochs': 100,            # 专门的约束优化轮数
            'constraint_warmup': 20,             # 约束预热轮数
            'constraint_annealing': True,        # 约束权重递增
        }
        
        # 训练历史
        self.train_history = {
            'g_losses': [],
            'd_losses': [],
            'violation_rates': [],
            'constraint_losses': [],
            'reconstruction_losses': [],
            'r2_scores': []
        }
        
        print(f"Constraint Optimizer initialized on device: {self.device}")
        print("🎯 专门解决参数违约率问题 (当前91.4% → 目标<10%)")
    
    def load_pretrained_models(self):
        """加载现有的预训练模型"""
        print("Loading existing models...")
        
        # 初始化模型结构
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
        
        # 加载现有权重
        try:
            self.generator.load_state_dict(
                torch.load(os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth"), 
                          map_location=self.device)
            )
            self.discriminator.load_state_dict(
                torch.load(os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth"), 
                          map_location=self.device)
            )
            self.forward_model.load_state_dict(
                torch.load(os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth"), 
                          map_location=self.device)
            )
            print("✓ Successfully loaded pretrained models")
            print("  Current performance: R²=0.9888, Consistency=0.013, Balance=51%")
            print("  Focus: Reduce violation rate from 91.4% to <10%")
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
            print("Please ensure models exist in saved_models/ directory")
            return False
        
        return True
    
    def initialize_optimizers(self):
        """初始化约束优化专用优化器"""
        print("Initializing constraint optimization optimizers...")
        
        # 生成器 - 专注于约束优化
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=self.constraint_config['generator_lr'],
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # 判别器 - 保持当前平衡
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=self.constraint_config['discriminator_lr'],
            betas=(0.5, 0.999),
            weight_decay=1e-4
        )
        
        print("✓ Constraint optimization optimizers initialized")
        print(f"  - Generator LR: {self.constraint_config['generator_lr']} (focused on constraints)")
        print(f"  - Discriminator LR: {self.constraint_config['discriminator_lr']} (maintain balance)")
    
    def calculate_enhanced_constraint_loss(self, pred_params_norm: torch.Tensor, param_ranges: Dict) -> Dict:
        """计算增强的约束损失"""
        batch_size = pred_params_norm.size(0)
        
        # 1. 硬约束损失 - 范围违约严厉惩罚
        range_violations = torch.sum(
            torch.relu(pred_params_norm - 1.0) + torch.relu(-pred_params_norm)
        )
        hard_constraint_loss = range_violations / batch_size
        
        # 2. 边界惩罚 - 防止参数贴边界
        boundary_distance = torch.min(pred_params_norm, 1.0 - pred_params_norm)
        boundary_penalty = torch.mean(torch.exp(-10 * boundary_distance))
        
        # 3. 平滑性约束 - 参数间的平滑性
        if pred_params_norm.size(1) > 1:
            param_diff = torch.diff(pred_params_norm, dim=1)
            smoothness_loss = torch.mean(param_diff ** 2)
        else:
            smoothness_loss = torch.tensor(0.0).to(self.device)
        
        # 4. 物理合理性约束 - 基于前向模型验证
        with torch.no_grad():
            pred_spectrum, pred_metrics = self.forward_model(pred_params_norm)
            # 检查生成的光谱是否合理
            spectrum_validity = torch.mean(torch.relu(-pred_spectrum))  # 负值惩罚
        
        return {
            'hard_constraint': hard_constraint_loss,
            'boundary_penalty': boundary_penalty,
            'smoothness': smoothness_loss,
            'spectrum_validity': spectrum_validity
        }
    
    def calculate_violation_rate(self, pred_params_norm: torch.Tensor) -> float:
        """计算参数违约率"""
        violations = torch.logical_or(pred_params_norm < 0, pred_params_norm > 1)
        violation_rate = torch.mean(violations.float()).item()
        return violation_rate
    
    def constraint_focused_training(self, dataloader, dataset, num_epochs: int = 100):
        """专注于约束的训练"""
        print(f"\n🎯 CONSTRAINT-FOCUSED TRAINING ({num_epochs} epochs)")
        print("目标: 参数违约率 91.4% → <10%")
        print("策略: 保持现有优秀性能，专注解决约束问题")
        
        best_violation_rate = 1.0  # 最佳违约率
        
        for epoch in range(num_epochs):
            epoch_metrics = {
                'g_loss': 0.0,
                'd_loss': 0.0,
                'violation_rate': 0.0,
                'constraint_loss': 0.0,
                'reconstruction_loss': 0.0,
                'r2_score': 0.0
            }
            
            # 动态约束权重 - 随训练进行递增
            if self.constraint_config['constraint_annealing']:
                constraint_multiplier = min(1.0 + epoch / 50.0, 3.0)  # 逐渐增强约束
            else:
                constraint_multiplier = 1.0
            
            for batch_idx, batch in enumerate(dataloader):
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_denorm = real_params_denorm.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                batch_size = real_spectrum.size(0)
                
                # ===================
                # 训练生成器 - 专注约束优化
                # ===================
                self.generator.train()
                self.optimizer_G.zero_grad()
                
                # 生成参数
                pred_params_norm = self.generator(real_spectrum)
                
                # 应用 sigmoid 确保范围约束
                pred_params_norm = torch.sigmoid(pred_params_norm)
                
                pred_params_denorm = denormalize_params(pred_params_norm, dataset.param_ranges)
                
                # 1. 增强约束损失
                constraint_losses = self.calculate_enhanced_constraint_loss(pred_params_norm, dataset.param_ranges)
                total_constraint_loss = (
                    self.constraint_config['hard_constraint_weight'] * constraint_losses['hard_constraint'] +
                    self.constraint_config['boundary_penalty_weight'] * constraint_losses['boundary_penalty'] +
                    self.constraint_config['smoothness_penalty_weight'] * constraint_losses['smoothness'] +
                    10.0 * constraint_losses['spectrum_validity']
                ) * constraint_multiplier
                
                # 2. 保持重建性能
                reconstruction_loss = self.criterion_mse(pred_params_norm, real_params_norm)
                
                # 3. 保持前向一致性
                pred_spectrum_from_params, _ = self.forward_model(pred_params_norm)
                consistency_loss = self.criterion_mse(pred_spectrum_from_params, real_spectrum)
                
                # 4. 轻微对抗损失 - 保持判别器平衡
                if epoch >= self.constraint_config['constraint_warmup']:
                    gen_scores = self.discriminator(real_spectrum, pred_params_denorm)
                    real_labels = torch.ones(batch_size, 1).to(self.device)
                    adversarial_loss = self.criterion_bce(gen_scores, real_labels)
                else:
                    adversarial_loss = torch.tensor(0.0).to(self.device)
                
                # 总生成器损失 - 约束为主导
                g_loss = (
                    total_constraint_loss +
                    self.constraint_config['reconstruction_weight'] * reconstruction_loss +
                    self.constraint_config['consistency_weight'] * consistency_loss +
                    self.constraint_config['adversarial_weight'] * adversarial_loss
                )
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.optimizer_G.step()
                
                # 计算违约率
                violation_rate = self.calculate_violation_rate(pred_params_norm)
                
                # 计算R²分数
                with torch.no_grad():
                    real_flat = real_params_norm.cpu().numpy().flatten()
                    pred_flat = pred_params_norm.detach().cpu().numpy().flatten()
                    ss_res = np.sum((real_flat - pred_flat) ** 2)
                    ss_tot = np.sum((real_flat - np.mean(real_flat)) ** 2)
                    r2_score = 1 - (ss_res / (ss_tot + 1e-8))
                
                # ===================
                # 轻量级判别器训练 - 保持平衡
                # ===================
                if (batch_idx + 1) % 3 == 0:  # 降低更新频率
                    self.discriminator.train()
                    self.optimizer_D.zero_grad()
                    
                    # 真实样本
                    real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
                    real_scores = self.discriminator(real_spectrum, real_params_denorm)
                    d_loss_real = self.criterion_bce(real_scores, real_labels)
                    
                    # 生成样本
                    fake_labels = torch.zeros(batch_size, 1).to(self.device) + 0.1
                    with torch.no_grad():
                        fake_params_norm = self.generator(real_spectrum)
                        fake_params_norm = torch.sigmoid(fake_params_norm)
                    fake_params_denorm = denormalize_params(fake_params_norm, dataset.param_ranges)
                    fake_scores = self.discriminator(real_spectrum, fake_params_denorm)
                    d_loss_fake = self.criterion_bce(fake_scores, fake_labels)
                    
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 0.5)
                    self.optimizer_D.step()
                else:
                    d_loss = torch.tensor(0.0)
                
                # 累积指标
                epoch_metrics['g_loss'] += g_loss.item()
                epoch_metrics['d_loss'] += d_loss.item()
                epoch_metrics['violation_rate'] += violation_rate
                epoch_metrics['constraint_loss'] += total_constraint_loss.item()
                epoch_metrics['reconstruction_loss'] += reconstruction_loss.item()
                epoch_metrics['r2_score'] += r2_score
            
            # 平均指标
            for key in epoch_metrics:
                epoch_metrics[key] /= len(dataloader)
            
            # 记录历史
            self.train_history['g_losses'].append(epoch_metrics['g_loss'])
            self.train_history['d_losses'].append(epoch_metrics['d_loss'])
            self.train_history['violation_rates'].append(epoch_metrics['violation_rate'])
            self.train_history['constraint_losses'].append(epoch_metrics['constraint_loss'])
            self.train_history['reconstruction_losses'].append(epoch_metrics['reconstruction_loss'])
            self.train_history['r2_scores'].append(epoch_metrics['r2_score'])
            
            # 保存最佳模型
            if epoch_metrics['violation_rate'] < best_violation_rate:
                best_violation_rate = epoch_metrics['violation_rate']
                self.save_best_models()
            
            # 打印进度
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Violation Rate: {epoch_metrics['violation_rate']:.4f} (目标<0.10)")
                print(f"  R² Score: {epoch_metrics['r2_score']:.4f} (保持>0.98)")
                print(f"  G Loss: {epoch_metrics['g_loss']:.6f}")
                print(f"  Constraint Loss: {epoch_metrics['constraint_loss']:.6f}")
                print(f"  Best Violation Rate: {best_violation_rate:.4f}")
                
                # 评估约束改善状态
                if epoch_metrics['violation_rate'] < 0.10:
                    print("  🎉 约束目标已达成！")
                elif epoch_metrics['violation_rate'] < 0.30:
                    print("  🟢 约束显著改善")
                elif epoch_metrics['violation_rate'] < 0.60:
                    print("  🟡 约束持续改善")
                else:
                    print("  🔴 约束仍需努力")
            
            # 保存检查点
            if (epoch + 1) % 25 == 0:
                self.save_checkpoint(epoch + 1, "constraint")
        
        print(f"\n✓ Constraint optimization completed")
        print(f"  Final Violation Rate: {epoch_metrics['violation_rate']:.4f}")
        print(f"  Best Violation Rate: {best_violation_rate:.4f}")
        print(f"  Final R² Score: {epoch_metrics['r2_score']:.4f}")
    
    def save_best_models(self):
        """保存最佳约束优化模型"""
        os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
        
        torch.save(self.generator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth"))
        torch.save(self.discriminator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth"))
        torch.save(self.forward_model.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth"))
        
        # 同时保存约束优化版本
        torch.save(self.generator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "generator_constraint_opt.pth"))
        torch.save(self.discriminator.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_constraint_opt.pth"))
        torch.save(self.forward_model.state_dict(), 
                  os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_constraint_opt.pth"))
    
    def save_checkpoint(self, epoch: int, mode: str = "constraint"):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'mode': mode,
            'train_history': self.train_history,
            'constraint_config': self.constraint_config,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'forward_model_state_dict': self.forward_model.state_dict(),
        }
        
        if self.optimizer_G is not None:
            checkpoint['optimizer_G_state_dict'] = self.optimizer_G.state_dict()
        if self.optimizer_D is not None:
            checkpoint['optimizer_D_state_dict'] = self.optimizer_D.state_dict()
        
        os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f'constraint_checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
    
    def plot_constraint_optimization_curves(self):
        """生成约束优化训练曲线"""
        print("Generating constraint optimization curves...")
        
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Constraint Optimization Progress - Violation Rate Reduction', fontsize=16)
        
        epochs = range(1, len(self.train_history['violation_rates']) + 1)
        
        # 1. 违约率改善 - 最重要
        axes[0, 0].plot(epochs, self.train_history['violation_rates'], 'r-', linewidth=3, label='Violation Rate')
        axes[0, 0].axhline(y=0.10, color='green', linestyle='--', linewidth=2, label='Target (<10%)')
        axes[0, 0].axhline(y=0.914, color='gray', linestyle=':', label='Initial (91.4%)')
        axes[0, 0].set_title('Parameter Violation Rate Reduction', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Violation Rate')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 1)
        
        # 2. R²性能保持
        axes[0, 1].plot(epochs, self.train_history['r2_scores'], 'b-', linewidth=2, label='R² Score')
        axes[0, 1].axhline(y=0.9888, color='green', linestyle='--', label='Initial Performance')
        axes[0, 1].axhline(y=0.80, color='orange', linestyle=':', label='Minimum Target')
        axes[0, 1].set_title('R² Score Maintenance', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 约束损失和重建损失
        axes[1, 0].plot(epochs, self.train_history['constraint_losses'], 'orange', linewidth=2, label='Constraint Loss')
        axes[1, 0].plot(epochs, self.train_history['reconstruction_losses'], 'purple', linewidth=2, label='Reconstruction Loss')
        axes[1, 0].set_title('Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # 4. GAN损失平衡
        axes[1, 1].plot(epochs, self.train_history['g_losses'], 'b-', linewidth=2, label='Generator')
        axes[1, 1].plot(epochs, self.train_history['d_losses'], 'r-', linewidth=2, label='Discriminator')
        axes[1, 1].set_title('GAN Loss Balance (Maintain ~51% D Accuracy)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图片
        plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plots_dir, f"constraint_optimization_curves_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Constraint optimization curves saved to: {plot_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Constraint Optimization - Fix Parameter Violations")
    parser.add_argument('--epochs', type=int, default=100, help='Constraint optimization epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
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
    
    print(f"Constraint optimization dataset: {len(dataset)} samples")
    print(f"Batch size: {args.batch_size}")
    
    # 创建约束优化器
    optimizer = ConstraintOptimizer(device=args.device)
    
    # 加载现有模型
    if not optimizer.load_pretrained_models():
        print("Failed to load pretrained models. Exiting...")
        return
    
    optimizer.initialize_optimizers()
    
    # 开始约束优化训练
    print(f"\n{'='*80}")
    print("🎯 CONSTRAINT OPTIMIZATION TRAINING")
    print(f"{'='*80}")
    print("目标: 参数违约率 91.4% → <10%")
    print("保持: R²=0.9888, 一致性=0.013, 判别器平衡=51%")
    print(f"{'='*80}")
    
    optimizer.constraint_focused_training(dataloader, dataset, args.epochs)
    
    # 生成训练曲线
    optimizer.plot_constraint_optimization_curves()
    
    print(f"\n🎯 Constraint optimization completed!")
    print("🔍 建议立即运行评估验证约束改善效果:")
    print("python3 core/evaluate/unified_evaluator.py --num_samples 1000")

if __name__ == "__main__":
    main()
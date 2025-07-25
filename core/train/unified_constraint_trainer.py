# PI_GAN_THZ/core/train/unified_constraint_trainer.py
# 统一约束训练器 - 整合所有训练功能，重点关注约束优化和应急修复策略

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, LinearLR
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List, Union
from tqdm import tqdm

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入所有模型
from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.models.forward_model import ForwardModel

# 导入所有需要的工具函数和损失函数
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_bce, criterion_mse, \
                             maxwell_equation_loss, lc_model_approx_loss, \
                             structural_param_range_loss, bnn_kl_loss

# 导入配置
import config.config as cfg
from config.training_optimization import get_optimization_config

class UnifiedConstraintTrainer:
    """
    统一约束训练器 - 整合前向模型预训练、PI-GAN训练、约束优化和应急修复策略
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
            'r2_scores': [],          # R²分数
            'detailed_losses': {      # 详细损失组件
                'adversarial': [],
                'reconstruction': [],
                'constraint': [],
                'physics': [],
                'cycle_consistency': []
            },
            'lr_history': {           # 学习率历史
                'generator': [],
                'discriminator': [],
                'forward_model': []
            }
        }
        
        # 约束配置
        self.constraint_config = {
            'hard_constraint_weight': 50.0,  # 硬约束权重
            'boundary_penalty_weight': 20.0,  # 边界惩罚权重
            'range_violation_penalty': 100.0,  # 范围违反惩罚
            'smoothness_penalty': 10.0,  # 平滑性惩罚
            'physics_constraint_weight': 30.0,  # 物理约束权重
            'max_constraint_multiplier': 10.0,  # 最大约束乘数
            'annealing_epochs': 50  # 退火轮数
        }
        
        # 应急配置
        self.emergency_config = {
            'forward_intensive_epochs': 200,  # 前向网络密集训练轮数
            'forward_lr': 1e-3,  # 前向网络学习率
            'gan_balanced_epochs': 200,  # GAN平衡训练轮数
            'discriminator_update_freq': 3,  # 判别器更新频率
            'warmup_epochs': 20,  # 预热轮数
            'label_smoothing': 0.1,  # 标签平滑
            'cycle_consistency_weight': 10.0,  # 循环一致性权重
            'l1_penalty_weight': 1.0  # L1惩罚权重
        }
        
        print(f"统一约束训练器初始化完成，使用设备: {self.device}")
    
    def initialize_models(self):
        """初始化所有模型"""
        print("初始化模型...")
        
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
        
        print("✓ 所有模型初始化成功")
        
        # 打印模型参数数量
        total_params = 0
        for name, model in [('Generator', self.generator), 
                           ('Discriminator', self.discriminator), 
                           ('Forward Model', self.forward_model)]:
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params += params
            print(f"  - {name}: {params:,} 参数")
        print(f"  - 总计: {total_params:,} 参数")
    
    def initialize_optimizers(self, mode: str = "standard", lr_g: float = None, lr_d: float = None, lr_f: float = None):
        """初始化优化器和调度器"""
        print(f"初始化优化器，模式: {mode}...")
        
        # 根据模式设置学习率
        if mode == "standard" or mode == "pigan_only":
            # 标准模式使用配置文件中的学习率
            lr_g = lr_g or cfg.LR_G
            lr_d = lr_d or cfg.LR_D
            lr_f = lr_f or cfg.FWD_PRETRAIN_LR
        elif mode == "constraint" or mode == "constraint_only":
            # 约束优化模式使用较高的生成器学习率
            lr_g = lr_g or 2e-4
            lr_d = lr_d or 5e-5  # 降低判别器学习率以平衡训练
            lr_f = lr_f or 1e-4
        elif mode == "emergency" or mode == "emergency_only":
            # 应急模式使用特殊学习率
            lr_g = lr_g or 1e-4
            lr_d = lr_d or 2e-5  # 大幅降低判别器学习率
            lr_f = lr_f or 1e-3  # 提高前向模型学习率
        elif mode == "forward_only" or mode == "progressive":
            # 前向模型训练模式
            lr_g = lr_g or cfg.LR_G
            lr_d = lr_d or cfg.LR_D
            lr_f = lr_f or cfg.FWD_PRETRAIN_LR
        
        # 创建优化器
        self.optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=lr_g,
            betas=(0.5, 0.999),
            weight_decay=1e-5
        )
        
        self.optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=lr_d,
            betas=(0.5, 0.999),
            weight_decay=1e-5
        )
        
        self.optimizer_F = optim.Adam(
            self.forward_model.parameters(),
            lr=lr_f,
            betas=(0.9, 0.999),
            weight_decay=1e-5
        )
        
        # 创建学习率调度器
        if mode == "standard" or mode == "pigan_only":
            self.scheduler_G = CosineAnnealingLR(self.optimizer_G, T_max=200, eta_min=lr_g * 0.1)
            self.scheduler_D = CosineAnnealingLR(self.optimizer_D, T_max=200, eta_min=lr_d * 0.1)
            self.scheduler_F = CosineAnnealingLR(self.optimizer_F, T_max=100, eta_min=lr_f * 0.1)
        elif mode == "constraint" or mode == "constraint_only":
            # 约束优化使用线性学习率调度
            self.scheduler_G = LinearLR(self.optimizer_G, start_factor=1.0, end_factor=0.5, total_iters=100)
            self.scheduler_D = LinearLR(self.optimizer_D, start_factor=1.0, end_factor=0.5, total_iters=100)
            self.scheduler_F = LinearLR(self.optimizer_F, start_factor=1.0, end_factor=0.5, total_iters=100)
        elif mode == "emergency" or mode == "emergency_only":
            # 应急模式使用步进学习率调度
            self.scheduler_G = StepLR(self.optimizer_G, step_size=50, gamma=0.5)
            self.scheduler_D = StepLR(self.optimizer_D, step_size=50, gamma=0.5)
            self.scheduler_F = StepLR(self.optimizer_F, step_size=50, gamma=0.5)
        elif mode == "forward_only" or mode == "progressive":
            # 前向模型训练模式使用步进学习率调度
            self.scheduler_G = StepLR(self.optimizer_G, step_size=50, gamma=0.5)
            self.scheduler_D = StepLR(self.optimizer_D, step_size=50, gamma=0.5)
            self.scheduler_F = StepLR(self.optimizer_F, step_size=50, gamma=0.5)
        
        print(f"✓ 优化器初始化完成")
        print(f"  - Generator LR: {lr_g}")
        print(f"  - Discriminator LR: {lr_d}")
        print(f"  - Forward Model LR: {lr_f}")
    
    def train_forward_model(self, dataloader, num_epochs: int = 100):
        """预训练前向模型"""
        print(f"\n=== 前向模型预训练 ({num_epochs} 轮) ===\n")
        
        self.forward_model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_spectrum_loss = 0.0
            epoch_metrics_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
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
                    5.0 * spectrum_loss +
                    2.0 * metrics_loss +
                    0.5 * smoothness_loss
                )
                
                # 反向传播
                self.optimizer_F.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
                self.optimizer_F.step()
                
                epoch_loss += total_loss.item()
                epoch_spectrum_loss += spectrum_loss.item()
                epoch_metrics_loss += metrics_loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'spec_loss': spectrum_loss.item(),
                    'metrics_loss': metrics_loss.item()
                })
            
            # 更新调度器
            self.scheduler_F.step()
            
            # 计算平均损失
            avg_loss = epoch_loss / len(dataloader)
            avg_spectrum_loss = epoch_spectrum_loss / len(dataloader)
            avg_metrics_loss = epoch_metrics_loss / len(dataloader)
            
            # 记录历史
            self.train_history['forward_losses'].append(avg_loss)
            self.train_history['lr_history']['forward_model'].append(self.scheduler_F.get_last_lr()[0])
            
            # 打印轮次总结
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"Loss: {avg_loss:.6f}, "
                  f"Spectrum: {avg_spectrum_loss:.6f}, "
                  f"Metrics: {avg_metrics_loss:.6f}, "
                  f"LR: {self.scheduler_F.get_last_lr()[0]:.6f}")
        
        print("✓ 前向模型预训练完成")
    
    def calculate_enhanced_constraint_loss(self, pred_params_norm: torch.Tensor, dataset) -> Tuple[torch.Tensor, float]:
        """计算增强的约束损失"""
        batch_size = pred_params_norm.size(0)
        param_dim = pred_params_norm.size(1)
        
        # 1. 硬约束：参数范围违约惩罚
        # 检测超出[0,1]范围的参数
        out_of_range = torch.max(
            torch.zeros_like(pred_params_norm),
            torch.max(
                pred_params_norm - 1.0,
                -pred_params_norm
            )
        )
        hard_constraint_loss = torch.sum(out_of_range ** 2) / batch_size
        
        # 2. 边界惩罚：靠近边界的平滑惩罚
        boundary_dist = torch.min(
            pred_params_norm,
            1.0 - pred_params_norm
        )
        boundary_penalty = torch.sum(
            torch.exp(-20 * boundary_dist)
        ) / batch_size
        
        # 3. 平滑性惩罚：相邻参数的平滑性
        param_diff = torch.diff(pred_params_norm, dim=1)
        smoothness_penalty = torch.mean(torch.abs(param_diff))
        
        # 4. 物理有效性检查：使用前向模型验证
        with torch.no_grad():
            pred_spectrum, _ = self.forward_model(pred_params_norm)
            # 检查生成的频谱是否有效（无NaN、无Inf、在合理范围内）
            invalid_spectrum = torch.logical_or(
                torch.isnan(pred_spectrum),
                torch.isinf(pred_spectrum)
            )
            spectrum_validity_loss = torch.sum(invalid_spectrum.float()) / batch_size
        
        # 计算总约束损失
        total_constraint_loss = (
            self.constraint_config['hard_constraint_weight'] * hard_constraint_loss +
            self.constraint_config['boundary_penalty_weight'] * boundary_penalty +
            self.constraint_config['smoothness_penalty'] * smoothness_penalty +
            self.constraint_config['physics_constraint_weight'] * spectrum_validity_loss
        )
        
        # 计算违约率
        with torch.no_grad():
            violations = torch.sum((pred_params_norm < 0) | (pred_params_norm > 1), dim=1)
            violation_rate = torch.mean((violations > 0).float()).item()
        
        return total_constraint_loss, violation_rate
    
    def calculate_r2_score(self, pred_params_norm: torch.Tensor, real_params_norm: torch.Tensor) -> float:
        """计算R²分数"""
        with torch.no_grad():
            # 计算总平方和 (TSS)
            mean_real = torch.mean(real_params_norm, dim=0, keepdim=True)
            tss = torch.sum((real_params_norm - mean_real) ** 2)
            
            # 计算残差平方和 (RSS)
            rss = torch.sum((real_params_norm - pred_params_norm) ** 2)
            
            # 计算R²
            r2 = 1 - (rss / tss)
            
            return r2.item()
    
    def train_pigan_standard(self, dataloader, dataset, num_epochs: int = 200):
        """标准PI-GAN训练"""
        print(f"\n=== 标准PI-GAN训练 ({num_epochs} 轮) ===\n")
        
        self.initialize_optimizers("standard")
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_violation_rate = 0.0
            epoch_r2 = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in progress_bar:
                real_spectrum, real_params_denorm, real_params_norm, _, real_metrics_norm = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_denorm = real_params_denorm.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                real_metrics_norm = real_metrics_norm.to(self.device)
                
                batch_size = real_spectrum.size(0)
                
                # =================== 训练判别器 ===================
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
                
                # 前向一致性损失
                with torch.no_grad():
                    pred_spectrum, pred_metrics = self.forward_model(pred_params_norm)
                g_loss_forward = self.criterion_mse(pred_spectrum, real_spectrum)
                
                # 参数范围约束损失
                g_loss_param_range = structural_param_range_loss(pred_params_norm)
                
                # 物理约束损失
                frequencies_tensor = torch.tensor(dataset.frequencies, dtype=torch.float32, device=self.device).unsqueeze(0)
                g_loss_maxwell = maxwell_equation_loss(pred_spectrum, frequencies_tensor, pred_params_norm)
                
                # 总生成器损失
                g_loss = (
                    1.0 * g_loss_adv +
                    5.0 * g_loss_recon +
                    2.0 * g_loss_forward +
                    5.0 * g_loss_param_range +
                    2.0 * g_loss_maxwell
                )
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.optimizer_G.step()
                
                # 计算约束违约率
                with torch.no_grad():
                    violations = torch.sum((pred_params_norm < 0) | (pred_params_norm > 1), dim=1)
                    violation_rate = torch.mean((violations > 0).float()).item()
                
                # 计算R²分数
                r2 = self.calculate_r2_score(pred_params_norm, real_params_norm)
                
                # 累积统计
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()
                epoch_violation_rate += violation_rate
                epoch_r2 += r2
                
                # 更新进度条
                progress_bar.set_postfix({
                    'G_loss': g_loss.item(),
                    'D_loss': d_loss.item(),
                    'Viol': violation_rate,
                    'R²': r2
                })
            
            # 更新调度器
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # 计算平均值
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)
            avg_violation_rate = epoch_violation_rate / len(dataloader)
            avg_r2 = epoch_r2 / len(dataloader)
            
            # 记录历史
            self.train_history['g_losses'].append(avg_g_loss)
            self.train_history['d_losses'].append(avg_d_loss)
            self.train_history['constraint_violations'].append(avg_violation_rate)
            self.train_history['r2_scores'].append(avg_r2)
            self.train_history['lr_history']['generator'].append(self.scheduler_G.get_last_lr()[0])
            self.train_history['lr_history']['discriminator'].append(self.scheduler_D.get_last_lr()[0])
            
            # 打印轮次总结
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"G_loss: {avg_g_loss:.4f}, "
                  f"D_loss: {avg_d_loss:.4f}, "
                  f"Violation: {avg_violation_rate:.4f}, "
                  f"R²: {avg_r2:.4f}, "
                  f"G_LR: {self.scheduler_G.get_last_lr()[0]:.6f}")
            
            # 保存检查点
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch + 1, "standard")
        
        print("✓ 标准PI-GAN训练完成")
    
    def constraint_focused_training(self, dataloader, dataset, num_epochs: int = 100):
        """约束优化训练"""
        print(f"\n=== 约束优化训练 ({num_epochs} 轮) ===\n")
        print(f"目标: 将参数违约率从 {self.train_history['constraint_violations'][-1]:.4f} 降低到 0.1 以下")
        
        self.initialize_optimizers("constraint")
        
        # 保存最佳模型的状态
        best_violation_rate = float('inf')
        best_model_state = None
        
        # 约束权重退火
        max_multiplier = self.constraint_config['max_constraint_multiplier']
        annealing_epochs = self.constraint_config['annealing_epochs']
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_constraint_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_violation_rate = 0.0
            epoch_r2 = 0.0
            
            # 计算当前约束乘数
            if epoch < annealing_epochs:
                constraint_multiplier = 1.0 + (max_multiplier - 1.0) * (epoch / annealing_epochs)
            else:
                constraint_multiplier = max_multiplier
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch in enumerate(progress_bar):
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_denorm = real_params_denorm.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                batch_size = real_spectrum.size(0)
                
                # =================== 训练生成器 ===================
                self.generator.train()
                self.optimizer_G.zero_grad()
                
                # 生成参数
                pred_params_norm = self.generator(real_spectrum)
                pred_params_denorm = denormalize_params(pred_params_norm, dataset.param_ranges)
                
                # 增强约束损失
                constraint_loss, violation_rate = self.calculate_enhanced_constraint_loss(pred_params_norm, dataset)
                
                # 重建损失
                recon_loss = self.criterion_mse(pred_params_norm, real_params_norm)
                
                # 前向一致性损失
                with torch.no_grad():
                    pred_spectrum, _ = self.forward_model(pred_params_norm)
                forward_loss = self.criterion_mse(pred_spectrum, real_spectrum)
                
                # 对抗损失 - 在约束训练中权重较低
                if batch_idx % 3 == 0:  # 减少判别器训练频率
                    self.discriminator.train()
                    self.optimizer_D.zero_grad()
                    
                    # 真实样本
                    real_labels = torch.ones(batch_size, 1).to(self.device) * 0.9
                    real_scores = self.discriminator(real_spectrum, real_params_denorm)
                    d_loss_real = self.criterion_bce(real_scores, real_labels)
                    
                    # 生成样本
                    fake_labels = torch.zeros(batch_size, 1).to(self.device) + 0.1
                    fake_scores = self.discriminator(real_spectrum, pred_params_denorm.detach())
                    d_loss_fake = self.criterion_bce(fake_scores, fake_labels)
                    
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.optimizer_D.step()
                    
                    # 生成器对抗损失
                    gen_scores = self.discriminator(real_spectrum, pred_params_denorm)
                    g_loss_adv = self.criterion_bce(gen_scores, real_labels)
                else:
                    g_loss_adv = torch.tensor(0.0, device=self.device)
                
                # 总生成器损失 - 约束优化阶段重点关注约束损失
                g_loss = (
                    0.5 * g_loss_adv +
                    2.0 * recon_loss +
                    1.0 * forward_loss +
                    constraint_multiplier * constraint_loss  # 动态增加约束权重
                )
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.optimizer_G.step()
                
                # 计算R²分数
                r2 = self.calculate_r2_score(pred_params_norm, real_params_norm)
                
                # 累积统计
                epoch_g_loss += g_loss.item()
                epoch_constraint_loss += constraint_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_violation_rate += violation_rate
                epoch_r2 += r2
                
                # 更新进度条
                progress_bar.set_postfix({
                    'G_loss': g_loss.item(),
                    'Constraint': constraint_loss.item(),
                    'Viol': violation_rate,
                    'R²': r2,
                    'Mult': constraint_multiplier
                })
            
            # 更新调度器
            self.scheduler_G.step()
            if hasattr(self, 'scheduler_D'):
                self.scheduler_D.step()
            
            # 计算平均值
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_constraint_loss = epoch_constraint_loss / len(dataloader)
            avg_recon_loss = epoch_recon_loss / len(dataloader)
            avg_violation_rate = epoch_violation_rate / len(dataloader)
            avg_r2 = epoch_r2 / len(dataloader)
            
            # 记录历史
            self.train_history['g_losses'].append(avg_g_loss)
            self.train_history['constraint_violations'].append(avg_violation_rate)
            self.train_history['r2_scores'].append(avg_r2)
            self.train_history['detailed_losses']['constraint'].append(avg_constraint_loss)
            self.train_history['detailed_losses']['reconstruction'].append(avg_recon_loss)
            self.train_history['lr_history']['generator'].append(self.scheduler_G.get_last_lr()[0])
            
            # 打印轮次总结
            print(f"Epoch [{epoch+1}/{num_epochs}] - "
                  f"G_loss: {avg_g_loss:.4f}, "
                  f"Constraint: {avg_constraint_loss:.4f}, "
                  f"Violation: {avg_violation_rate:.4f}, "
                  f"R²: {avg_r2:.4f}, "
                  f"Multiplier: {constraint_multiplier:.2f}")
            
            # 保存最佳模型
            if avg_violation_rate < best_violation_rate:
                best_violation_rate = avg_violation_rate
                best_model_state = {
                    'generator': self.generator.state_dict(),
                    'discriminator': self.discriminator.state_dict(),
                    'forward_model': self.forward_model.state_dict(),
                    'epoch': epoch + 1,
                    'violation_rate': best_violation_rate,
                    'r2_score': avg_r2
                }
                print(f"✓ 新的最佳违约率: {best_violation_rate:.4f}，已保存模型状态")
            
            # 保存检查点
            if (epoch + 1) % 20 == 0:
                self.save_checkpoint(epoch + 1, "constraint")
            
            # 提前停止条件
            if avg_violation_rate < 0.1:
                print(f"✓ 已达到目标违约率 ({avg_violation_rate:.4f} < 0.1)，提前停止训练")
                break
        
        # 恢复最佳模型
        if best_model_state is not None:
            self.generator.load_state_dict(best_model_state['generator'])
            self.discriminator.load_state_dict(best_model_state['discriminator'])
            self.forward_model.load_state_dict(best_model_state['forward_model'])
            print(f"✓ 已恢复最佳模型 (Epoch {best_model_state['epoch']}, "
                  f"Violation: {best_model_state['violation_rate']:.4f}, "
                  f"R²: {best_model_state['r2_score']:.4f})")
        
        print("✓ 约束优化训练完成")
    
    def emergency_repair_training(self, dataloader, dataset, num_epochs_forward: int = 200, num_epochs_gan: int = 200):
        """应急修复训练"""
        print(f"\n=== 应急修复训练 ===\n")
        print(f"阶段1: 前向网络密集训练 ({num_epochs_forward} 轮)")
        print(f"阶段2: 平衡GAN训练 ({num_epochs_gan} 轮)")
        
        # 初始化应急优化器
        self.initialize_optimizers("emergency")
        
        # =================== 阶段1: 前向网络密集训练 ===================
        print("\n开始前向网络密集训练...")
        
        best_forward_loss = float('inf')
        best_forward_state = None
        early_stop_counter = 0
        
        self.forward_model.train()
        
        for epoch in range(num_epochs_forward):
            epoch_loss = 0.0
            epoch_spectrum_loss = 0.0
            epoch_metrics_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"前向训练 {epoch+1}/{num_epochs_forward}")
            for batch in progress_bar:
                real_spectrum, _, real_params_norm, _, real_metrics_norm = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                real_metrics_norm = real_metrics_norm.to(self.device)
                
                # 前向传播
                pred_spectrum, pred_metrics = self.forward_model(real_params_norm)
                
                # 损失计算 - 使用MSE和L1的组合
                spectrum_mse = self.criterion_mse(pred_spectrum, real_spectrum)
                spectrum_l1 = torch.mean(torch.abs(pred_spectrum - real_spectrum))
                spectrum_loss = spectrum_mse + 0.5 * spectrum_l1
                
                metrics_mse = self.criterion_mse(pred_metrics, real_metrics_norm)
                metrics_l1 = torch.mean(torch.abs(pred_metrics - real_metrics_norm))
                metrics_loss = metrics_mse + 0.5 * metrics_l1
                
                # 平滑性损失
                spectrum_diff = torch.diff(pred_spectrum, dim=1)
                smoothness_loss = torch.mean(spectrum_diff ** 2)
                
                # 总损失 - 重新平衡权重
                total_loss = (
                    5.0 * spectrum_loss +
                    2.0 * metrics_loss +
                    1.0 * smoothness_loss
                )
                
                # 反向传播
                self.optimizer_F.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), 1.0)
                self.optimizer_F.step()
                
                epoch_loss += total_loss.item()
                epoch_spectrum_loss += spectrum_loss.item()
                epoch_metrics_loss += metrics_loss.item()
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': total_loss.item(),
                    'spec': spectrum_loss.item(),
                    'metrics': metrics_loss.item()
                })
            
            # 更新调度器
            self.scheduler_F.step()
            
            # 计算平均损失
            avg_loss = epoch_loss / len(dataloader)
            avg_spectrum_loss = epoch_spectrum_loss / len(dataloader)
            avg_metrics_loss = epoch_metrics_loss / len(dataloader)
            
            # 记录历史
            self.train_history['forward_losses'].append(avg_loss)
            self.train_history['lr_history']['forward_model'].append(self.scheduler_F.get_last_lr()[0])
            
            # 打印轮次总结
            print(f"前向训练 [{epoch+1}/{num_epochs_forward}] - "
                  f"Loss: {avg_loss:.6f}, "
                  f"Spectrum: {avg_spectrum_loss:.6f}, "
                  f"Metrics: {avg_metrics_loss:.6f}, "
                  f"LR: {self.scheduler_F.get_last_lr()[0]:.6f}")
            
            # 保存最佳模型
            if avg_loss < best_forward_loss:
                best_forward_loss = avg_loss
                best_forward_state = self.forward_model.state_dict()
                early_stop_counter = 0
                print(f"✓ 新的最佳前向模型损失: {best_forward_loss:.6f}")
            else:
                early_stop_counter += 1
            
            # 提前停止
            if early_stop_counter >= 20:
                print(f"✓ 前向模型训练20轮无改善，提前停止")
                break
        
        # 恢复最佳前向模型
        if best_forward_state is not None:
            self.forward_model.load_state_dict(best_forward_state)
            print(f"✓ 已恢复最佳前向模型 (Loss: {best_forward_loss:.6f})")
        
        # =================== 阶段2: 平衡GAN训练 ===================
        print("\n开始平衡GAN训练...")
        
        # 重新初始化GAN优化器，使用较低的学习率
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=5e-5, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=1e-5, betas=(0.5, 0.999))
        
        self.scheduler_G = StepLR(self.optimizer_G, step_size=50, gamma=0.5)
        self.scheduler_D = StepLR(self.optimizer_D, step_size=50, gamma=0.5)
        
        # 预热阶段轮数
        warmup_epochs = self.emergency_config['warmup_epochs']
        
        for epoch in range(num_epochs_gan):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            epoch_cycle_loss = 0.0
            epoch_r2 = 0.0
            epoch_violation_rate = 0.0
            
            # 确定当前是否在预热阶段
            is_warmup = epoch < warmup_epochs
            
            progress_bar = tqdm(dataloader, desc=f"GAN训练 {epoch+1}/{num_epochs_gan}")
            for batch_idx, batch in enumerate(progress_bar):
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_denorm = real_params_denorm.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                batch_size = real_spectrum.size(0)
                
                # =================== 训练判别器 ===================
                # 在预热阶段减少判别器训练频率
                train_discriminator = not is_warmup and (batch_idx % self.emergency_config['discriminator_update_freq'] == 0)
                
                if train_discriminator:
                    self.discriminator.train()
                    self.optimizer_D.zero_grad()
                    
                    # 真实样本
                    real_labels = torch.ones(batch_size, 1).to(self.device) * (1 - self.emergency_config['label_smoothing'])
                    real_scores = self.discriminator(real_spectrum, real_params_denorm)
                    d_loss_real = self.criterion_bce(real_scores, real_labels)
                    
                    # 生成样本
                    fake_labels = torch.zeros(batch_size, 1).to(self.device) + self.emergency_config['label_smoothing']
                    with torch.no_grad():
                        fake_params_norm = self.generator(real_spectrum)
                    fake_params_denorm = denormalize_params(fake_params_norm, dataset.param_ranges)
                    fake_scores = self.discriminator(real_spectrum, fake_params_denorm)
                    d_loss_fake = self.criterion_bce(fake_scores, fake_labels)
                    
                    d_loss = (d_loss_real + d_loss_fake) / 2
                    d_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
                    self.optimizer_D.step()
                else:
                    d_loss = torch.tensor(0.0, device=self.device)
                
                # =================== 训练生成器 ===================
                self.generator.train()
                self.optimizer_G.zero_grad()
                
                # 生成参数
                pred_params_norm = self.generator(real_spectrum)
                pred_params_denorm = denormalize_params(pred_params_norm, dataset.param_ranges)
                
                # 对抗损失 - 预热阶段不使用
                if not is_warmup:
                    gen_scores = self.discriminator(real_spectrum, pred_params_denorm)
                    g_loss_adv = self.criterion_bce(gen_scores, real_labels)
                else:
                    g_loss_adv = torch.tensor(0.0, device=self.device)
                
                # 重建损失
                g_loss_recon = self.criterion_mse(pred_params_norm, real_params_norm)
                
                # L1惩罚 - 促进稀疏性
                g_loss_l1 = torch.mean(torch.abs(pred_params_norm))
                
                # 循环一致性损失
                with torch.no_grad():
                    # 前向循环: 参数 -> 光谱
                    pred_spectrum, _ = self.forward_model(pred_params_norm)
                    # 反向循环: 光谱 -> 参数
                    cycle_params_norm = self.generator(pred_spectrum)
                
                cycle_consistency_loss = self.criterion_mse(cycle_params_norm, pred_params_norm)
                
                # 约束损失
                constraint_loss, violation_rate = self.calculate_enhanced_constraint_loss(pred_params_norm, dataset)
                
                # 总生成器损失 - 重新平衡权重
                g_loss = (
                    (0.0 if is_warmup else 1.0) * g_loss_adv +
                    5.0 * g_loss_recon +
                    self.emergency_config['l1_penalty_weight'] * g_loss_l1 +
                    self.emergency_config['cycle_consistency_weight'] * cycle_consistency_loss +
                    3.0 * constraint_loss
                )
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
                self.optimizer_G.step()
                
                # 计算R²分数
                r2 = self.calculate_r2_score(pred_params_norm, real_params_norm)
                
                # 累积统计
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item() if train_discriminator else 0.0
                epoch_cycle_loss += cycle_consistency_loss.item()
                epoch_r2 += r2
                epoch_violation_rate += violation_rate
                
                # 更新进度条
                progress_bar.set_postfix({
                    'G_loss': g_loss.item(),
                    'D_loss': d_loss.item() if train_discriminator else 0.0,
                    'Cycle': cycle_consistency_loss.item(),
                    'R²': r2,
                    'Viol': violation_rate,
                    'Mode': 'Warmup' if is_warmup else 'Full'
                })
            
            # 更新调度器
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # 计算平均值
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader) if not is_warmup else 0.0
            avg_cycle_loss = epoch_cycle_loss / len(dataloader)
            avg_r2 = epoch_r2 / len(dataloader)
            avg_violation_rate = epoch_violation_rate / len(dataloader)
            
            # 记录历史
            self.train_history['g_losses'].append(avg_g_loss)
            self.train_history['d_losses'].append(avg_d_loss)
            self.train_history['detailed_losses']['cycle_consistency'].append(avg_cycle_loss)
            self.train_history['r2_scores'].append(avg_r2)
            self.train_history['constraint_violations'].append(avg_violation_rate)
            self.train_history['lr_history']['generator'].append(self.scheduler_G.get_last_lr()[0])
            self.train_history['lr_history']['discriminator'].append(self.scheduler_D.get_last_lr()[0])
            
            # 打印轮次总结
            print(f"GAN训练 [{epoch+1}/{num_epochs_gan}] - "
                  f"G_loss: {avg_g_loss:.4f}, "
                  f"D_loss: {avg_d_loss:.4f}, "
                  f"Cycle: {avg_cycle_loss:.4f}, "
                  f"R²: {avg_r2:.4f}, "
                  f"Violation: {avg_violation_rate:.4f}, "
                  f"Mode: {'预热' if is_warmup else '完整'}")
            
            # 保存检查点
            if (epoch + 1) % 50 == 0:
                self.save_checkpoint(epoch + 1, "emergency")
        
        print("✓ 应急修复训练完成")
    
    def progressive_training_pipeline(self, dataloader, dataset):
        """渐进式训练流水线"""
        print(f"\n{'='*60}")
        print("渐进式训练流水线")
        print(f"{'='*60}")
        print("阶段1: 前向模型预训练 (100轮)")
        print("阶段2: 标准PI-GAN训练 (200轮)")
        print("阶段3: 约束优化训练 (100轮)")
        print("阶段4: 应急修复训练 (如需要)")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # 阶段1: 前向模型预训练
        self.train_forward_model(dataloader, num_epochs=100)
        
        # 阶段2: 标准PI-GAN训练
        self.train_pigan_standard(dataloader, dataset, num_epochs=200)
        
        # 检查违约率
        violation_rate = self.train_history['constraint_violations'][-1]
        print(f"\n当前参数违约率: {violation_rate:.4f}")
        
        # 阶段3: 约束优化训练 (如果违约率高)
        if violation_rate > 0.2:
            print("违约率过高，启动约束优化训练...")
            self.constraint_focused_training(dataloader, dataset, num_epochs=100)
        else:
            print("违约率可接受，跳过约束优化训练")
        
        # 检查R²分数
        r2_score = self.train_history['r2_scores'][-1]
        print(f"\n当前R²分数: {r2_score:.4f}")
        
        # 阶段4: 应急修复训练 (如果R²分数低)
        if r2_score < 0.7:
            print("R²分数过低，启动应急修复训练...")
            self.emergency_repair_training(dataloader, dataset, num_epochs_forward=100, num_epochs_gan=100)
        else:
            print("R²分数可接受，跳过应急修复训练")
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"渐进式训练完成，总耗时: {total_time:.2f}秒 ({total_time/60:.1f}分钟)")
        print(f"{'='*60}")
        
        # 生成训练曲线
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """生成训练曲线"""
        print("生成训练曲线...")
        
        # 设置中文绘图
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        fig.suptitle('PI-GAN训练进度', fontsize=16)
        
        # 1. 前向模型损失
        if self.train_history['forward_losses']:
            forward_epochs = range(1, len(self.train_history['forward_losses']) + 1)
            axes[0, 0].plot(forward_epochs, self.train_history['forward_losses'], 'b-', linewidth=2)
            axes[0, 0].set_title('前向模型训练损失')
            axes[0, 0].set_xlabel('轮次')
            axes[0, 0].set_ylabel('损失')
            axes[0, 0].grid(True, alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, '无前向模型训练数据', ha='center', va='center')
            axes[0, 0].set_title('前向模型训练损失')
        
        # 2. 生成器和判别器损失
        if self.train_history['g_losses']:
            gan_epochs = range(1, len(self.train_history['g_losses']) + 1)
            axes[0, 1].plot(gan_epochs, self.train_history['g_losses'], 'b-', label='生成器', linewidth=2)
            axes[0, 1].plot(gan_epochs, self.train_history['d_losses'], 'r-', label='判别器', linewidth=2)
            axes[0, 1].set_title('生成器与判别器损失')
            axes[0, 1].set_xlabel('轮次')
            axes[0, 1].set_ylabel('损失')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        else:
            axes[0, 1].text(0.5, 0.5, '无GAN训练数据', ha='center', va='center')
            axes[0, 1].set_title('生成器与判别器损失')
        
        # 3. 约束违约率
        if self.train_history['constraint_violations']:
            axes[1, 0].plot(gan_epochs, self.train_history['constraint_violations'], 'orange', linewidth=2)
            axes[1, 0].axhline(y=0.1, color='green', linestyle='--', label='目标 (10%)')
            axes[1, 0].set_title('参数约束违约率')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('违约率')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, '无违约率数据', ha='center', va='center')
            axes[1, 0].set_title('参数约束违约率')
        
        # 4. R²分数
        if self.train_history['r2_scores']:
            axes[1, 1].plot(gan_epochs, self.train_history['r2_scores'], 'g-', linewidth=2)
            axes[1, 1].axhline(y=0.8, color='blue', linestyle='--', label='目标 (0.8)')
            axes[1, 1].set_title('参数预测R²分数')
            axes[1, 1].set_xlabel('轮次')
            axes[1, 1].set_ylabel('R²')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, '无R²分数数据', ha='center', va='center')
            axes[1, 1].set_title('参数预测R²分数')
        
        # 5. 损失组件分解
        if any(self.train_history['detailed_losses'][k] for k in self.train_history['detailed_losses']):
            components = []
            for component in ['adversarial', 'reconstruction', 'constraint', 'physics', 'cycle_consistency']:
                if self.train_history['detailed_losses'][component]:
                    components.append(component)
            
            colors = ['blue', 'green', 'orange', 'purple', 'brown']
            for i, (component, color) in enumerate(zip(components, colors)):
                if self.train_history['detailed_losses'][component]:
                    # 确保长度匹配
                    comp_data = self.train_history['detailed_losses'][component]
                    comp_epochs = range(1, len(comp_data) + 1)
                    axes[2, 0].plot(comp_epochs, comp_data, color=color, label=component, linewidth=2)
            
            axes[2, 0].set_title('损失组件分解')
            axes[2, 0].set_xlabel('轮次')
            axes[2, 0].set_ylabel('损失值')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, '无详细损失数据', ha='center', va='center')
            axes[2, 0].set_title('损失组件分解')
        
        # 6. 学习率
        if any(self.train_history['lr_history'][k] for k in self.train_history['lr_history']):
            components = []
            for component in ['generator', 'discriminator', 'forward_model']:
                if self.train_history['lr_history'][component]:
                    components.append(component)
            
            colors = ['red', 'blue', 'green']
            for i, (component, color) in enumerate(zip(components, colors)):
                if self.train_history['lr_history'][component]:
                    # 确保长度匹配
                    lr_data = self.train_history['lr_history'][component]
                    lr_epochs = range(1, len(lr_data) + 1)
                    axes[2, 1].plot(lr_epochs, lr_data, color=color, label=component, linewidth=2)
            
            axes[2, 1].set_title('学习率调度')
            axes[2, 1].set_xlabel('轮次')
            axes[2, 1].set_ylabel('学习率')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, '无学习率数据', ha='center', va='center')
            axes[2, 1].set_title('学习率调度')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("✓ 训练曲线已保存至 'training_curves.png'")
        plt.close()
    
    def save_checkpoint(self, epoch: int, mode: str = "standard"):
        """保存检查点"""
        checkpoint_dir = os.path.join(project_root, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{mode}_epoch_{epoch}.pth')
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'forward_model_state_dict': self.forward_model.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'optimizer_F_state_dict': self.optimizer_F.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict() if self.scheduler_G else None,
            'scheduler_D_state_dict': self.scheduler_D.state_dict() if self.scheduler_D else None,
            'scheduler_F_state_dict': self.scheduler_F.state_dict() if self.scheduler_F else None,
            'train_history': self.train_history,
            'constraint_config': self.constraint_config,
            'emergency_config': self.emergency_config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ 检查点已保存至 '{checkpoint_path}'")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"错误: 检查点 '{checkpoint_path}' 不存在")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载模型状态
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
        
        # 加载优化器状态
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.optimizer_F.load_state_dict(checkpoint['optimizer_F_state_dict'])
        
        # 加载调度器状态
        if checkpoint['scheduler_G_state_dict'] and self.scheduler_G:
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
        if checkpoint['scheduler_D_state_dict'] and self.scheduler_D:
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
        if checkpoint['scheduler_F_state_dict'] and self.scheduler_F:
            self.scheduler_F.load_state_dict(checkpoint['scheduler_F_state_dict'])
        
        # 加载训练历史
        self.train_history = checkpoint['train_history']
        
        # 加载配置
        if 'constraint_config' in checkpoint:
            self.constraint_config = checkpoint['constraint_config']
        if 'emergency_config' in checkpoint:
            self.emergency_config = checkpoint['emergency_config']
        
        print(f"✓ 已从 '{checkpoint_path}' 加载检查点 (轮次 {checkpoint['epoch']})")
        return True
    
    def save_final_models(self, mode: str = "unified_constraint"):
        """保存最终模型"""
        models_dir = os.path.join(project_root, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # 保存生成器
        generator_path = os.path.join(models_dir, f'generator_{mode}.pth')
        torch.save(self.generator.state_dict(), generator_path)
        
        # 保存判别器
        discriminator_path = os.path.join(models_dir, f'discriminator_{mode}.pth')
        torch.save(self.discriminator.state_dict(), discriminator_path)
        
        # 保存前向模型
        forward_model_path = os.path.join(models_dir, f'forward_model_{mode}.pth')
        torch.save(self.forward_model.state_dict(), forward_model_path)
        
        # 保存训练历史
        history_path = os.path.join(models_dir, f'training_history_{mode}.pth')
        torch.save(self.train_history, history_path)
        
        print(f"✓ 最终模型已保存至 '{models_dir}'")
        print(f"  - 生成器: {generator_path}")
        print(f"  - 判别器: {discriminator_path}")
        print(f"  - 前向模型: {forward_model_path}")
        print(f"  - 训练历史: {history_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='统一约束训练器')
    parser.add_argument('--mode', type=str, default='progressive', choices=['forward_only', 'pigan_only', 'constraint_only', 'emergency_only', 'progressive'],
                        help='训练模式: forward_only, pigan_only, constraint_only, emergency_only, progressive')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--device', type=str, default='auto', help='设备: cuda, cpu, auto')
    parser.add_argument('--checkpoint', type=str, default=None, help='检查点路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建数据集和数据加载器
    import config.config as cfg
    dataset = MetamaterialDataset(data_path=cfg.DATASET_PATH)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"数据集大小: {len(dataset)}")
    print(f"批量大小: {args.batch_size}")
    print(f"批次数量: {len(dataloader)}")
    
    # 创建训练器
    trainer = UnifiedConstraintTrainer(device=args.device)
    trainer.initialize_models()
    trainer.initialize_optimizers(mode=args.mode)
    
    # 加载检查点
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # 根据模式执行训练
    if args.mode == 'forward_only':
        epochs = args.epochs or 200
        print(f"执行前向模型训练 ({epochs} 轮)...")
        trainer.train_forward_model(dataloader, num_epochs=epochs)
    
    elif args.mode == 'pigan_only':
        epochs = args.epochs or 300
        print(f"执行标准PI-GAN训练 ({epochs} 轮)...")
        trainer.train_pigan_standard(dataloader, dataset, num_epochs=epochs)
    
    elif args.mode == 'constraint_only':
        epochs = args.epochs or 100
        print(f"执行约束优化训练 ({epochs} 轮)...")
        trainer.constraint_focused_training(dataloader, dataset, num_epochs=epochs)
    
    elif args.mode == 'emergency_only':
        forward_epochs = args.epochs or 200
        gan_epochs = args.epochs or 200
        print(f"执行应急修复训练 (前向: {forward_epochs} 轮, GAN: {gan_epochs} 轮)...")
        trainer.emergency_repair_training(dataloader, dataset, num_epochs_forward=forward_epochs, num_epochs_gan=gan_epochs)
    
    elif args.mode == 'progressive':
        print("执行渐进式训练流水线...")
        trainer.progressive_training_pipeline(dataloader, dataset)
    
    # 保存最终模型
    trainer.save_final_models(mode=args.mode)
    
    # 生成训练曲线
    trainer.plot_training_curves()
    
    print("训练完成!")


if __name__ == "__main__":
    main()
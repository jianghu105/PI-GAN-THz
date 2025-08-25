"""
增强的训练流程模块
包含前向模型质量验证、训练监控、梯度累积等功能
专门针对小数据集优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from collections import defaultdict
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from config import config
from core.utils.data_loader import get_dataloaders
from core.models.forward_model import ForwardModel
from core.utils.loss import WeightedMSELoss
from core.utils.small_dataset_trainer import SmallDatasetTrainer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(config.LOG_DIR, 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ForwardModelValidator:
    """前向模型质量验证器"""
    
    def __init__(self, quality_thresholds: Dict[str, float] = None):
        """
        Args:
            quality_thresholds: 质量阈值字典
        """
        self.thresholds = quality_thresholds or {
            'spectra_mse': 0.05,      # 光谱MSE阈值
            'spectra_mae': 2.0,       # 光谱MAE阈值
            'metrics_mse': 0.02,      # 指标MSE阈值
            'metrics_mae': 0.15,      # 指标MAE阈值
            'r2_score': 0.85,         # R²得分阈值
            'correlation': 0.9        # 相关系数阈值
        }
        
    def validate_model_quality(self, model: nn.Module, val_loader, 
                             scalers: Dict[str, Any]) -> Dict[str, Any]:
        """验证前向模型质量"""
        logger.info("开始前向模型质量验证...")
        
        model.eval()
        all_predictions_spectra = []
        all_targets_spectra = []
        all_predictions_metrics = []
        all_targets_metrics = []
        
        # 准备缩放器张量
        metrics_scale = torch.tensor(scalers['metrics'].scale_, dtype=torch.float32).to(config.DEVICE)
        metrics_offset = torch.tensor(scalers['metrics'].min_, dtype=torch.float32).to(config.DEVICE)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证模型质量"):
                struct_params = batch['struct'].to(config.DEVICE)
                target_spectra = batch['spectra'].to(config.DEVICE)
                target_metrics = batch['metrics'].to(config.DEVICE)
                
                predicted_spectra, predicted_metrics = model(struct_params)
                
                # 收集光谱预测
                all_predictions_spectra.append(predicted_spectra.cpu())
                all_targets_spectra.append(target_spectra.cpu())
                
                # 处理指标预测（归一化到相同空间）
                predicted_metrics_cleaned = torch.nan_to_num(predicted_metrics, nan=0.0)
                predicted_metrics_scaled = predicted_metrics_cleaned * metrics_scale + metrics_offset
                
                all_predictions_metrics.append(predicted_metrics_scaled.cpu())
                all_targets_metrics.append(target_metrics.cpu())
        
        # 合并所有预测
        pred_spectra = torch.cat(all_predictions_spectra, dim=0).numpy()
        true_spectra = torch.cat(all_targets_spectra, dim=0).numpy()
        pred_metrics = torch.cat(all_predictions_metrics, dim=0).numpy()
        true_metrics = torch.cat(all_targets_metrics, dim=0).numpy()
        
        # 计算评估指标
        results = self._compute_metrics(pred_spectra, true_spectra, pred_metrics, true_metrics)
        
        # 质量检查
        quality_check = self._check_quality(results)
        results['quality_check'] = quality_check
        
        # 记录结果
        logger.info("前向模型质量验证结果:")
        logger.info(f"  光谱 MSE: {results['spectra_mse']:.6f} (阈值: {self.thresholds['spectra_mse']:.6f})")
        logger.info(f"  光谱 MAE: {results['spectra_mae']:.6f} (阈值: {self.thresholds['spectra_mae']:.6f})")
        logger.info(f"  指标 MSE: {results['metrics_mse']:.6f} (阈值: {self.thresholds['metrics_mse']:.6f})")
        logger.info(f"  指标 MAE: {results['metrics_mae']:.6f} (阈值: {self.thresholds['metrics_mae']:.6f})")
        logger.info(f"  整体质量: {'通过' if quality_check['overall_passed'] else '未通过'}")
        
        return results
    
    def _compute_metrics(self, pred_spectra, true_spectra, pred_metrics, true_metrics):
        """计算评估指标"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from scipy.stats import pearsonr
        
        results = {}
        
        # 光谱指标
        results['spectra_mse'] = mean_squared_error(true_spectra.flatten(), pred_spectra.flatten())
        results['spectra_mae'] = mean_absolute_error(true_spectra.flatten(), pred_spectra.flatten())
        results['spectra_r2'] = r2_score(true_spectra.flatten(), pred_spectra.flatten())
        
        # 指标指标
        results['metrics_mse'] = mean_squared_error(true_metrics.flatten(), pred_metrics.flatten())
        results['metrics_mae'] = mean_absolute_error(true_metrics.flatten(), pred_metrics.flatten())
        results['metrics_r2'] = r2_score(true_metrics.flatten(), pred_metrics.flatten())
        
        # 相关性分析
        spectra_corr, _ = pearsonr(true_spectra.flatten(), pred_spectra.flatten())
        metrics_corr, _ = pearsonr(true_metrics.flatten(), pred_metrics.flatten())
        
        results['spectra_correlation'] = spectra_corr
        results['metrics_correlation'] = metrics_corr
        
        # 按参数的详细分析
        results['per_metric_analysis'] = {}
        for i, metric_name in enumerate(config.METRIC_PARAMS):
            if i < true_metrics.shape[1]:
                metric_mse = mean_squared_error(true_metrics[:, i], pred_metrics[:, i])
                metric_r2 = r2_score(true_metrics[:, i], pred_metrics[:, i])
                results['per_metric_analysis'][metric_name] = {
                    'mse': metric_mse,
                    'r2': metric_r2
                }
        
        return results
    
    def _check_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """检查模型质量是否达标"""
        checks = {}
        
        # 各项指标检查
        checks['spectra_mse_passed'] = results['spectra_mse'] < self.thresholds['spectra_mse']
        checks['spectra_mae_passed'] = results['spectra_mae'] < self.thresholds['spectra_mae']
        checks['metrics_mse_passed'] = results['metrics_mse'] < self.thresholds['metrics_mse']
        checks['metrics_mae_passed'] = results['metrics_mae'] < self.thresholds['metrics_mae']
        checks['spectra_r2_passed'] = results['spectra_r2'] > self.thresholds['r2_score']
        checks['metrics_r2_passed'] = results['metrics_r2'] > self.thresholds['r2_score']
        checks['spectra_correlation_passed'] = results['spectra_correlation'] > self.thresholds['correlation']
        checks['metrics_correlation_passed'] = results['metrics_correlation'] > self.thresholds['correlation']
        
        # 整体评估
        critical_checks = [
            checks['spectra_mse_passed'],
            checks['metrics_mse_passed'],
            checks['spectra_r2_passed'],
            checks['metrics_r2_passed']
        ]
        
        checks['overall_passed'] = all(critical_checks)
        checks['passed_count'] = sum(checks.values() if isinstance(v, bool) for v in checks.values())
        checks['total_checks'] = len([v for v in checks.values() if isinstance(v, bool)])
        
        return checks

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir or config.LOG_DIR
        self.metrics_history = defaultdict(list)
        self.epoch_times = []
        self.learning_rates = []
        self.gradient_norms = []
        
        os.makedirs(self.save_dir, exist_ok=True)
        
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float], 
                         learning_rate: float = None, grad_norm: float = None):
        """记录epoch指标"""
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        if grad_norm is not None:
            self.gradient_norms.append(grad_norm)
        
        # 保存到JSON文件
        self._save_metrics()
    
    def _save_metrics(self):
        """保存指标到文件"""
        metrics_file = os.path.join(self.save_dir, 'training_metrics.json')
        
        data = {
            'metrics_history': dict(self.metrics_history),
            'learning_rates': self.learning_rates,
            'gradient_norms': self.gradient_norms
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def plot_training_curves(self, save_path: str = None):
        """绘制训练曲线"""
        if not self.metrics_history:
            logger.warning("没有训练历史可以绘制")
            return
        
        # 确定子图数量
        metrics_to_plot = ['train_loss', 'val_loss', 'train_spectra_loss', 'val_spectra_loss', 
                          'train_metrics_loss', 'val_metrics_loss']
        available_metrics = [m for m in metrics_to_plot if m in self.metrics_history]
        
        if not available_metrics:
            logger.warning("没有可绘制的指标")
            return
        
        n_plots = len(available_metrics) + (1 if self.learning_rates else 0) + (1 if self.gradient_norms else 0)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        plot_idx = 0
        
        # 绘制损失曲线
        for metric in available_metrics:
            row, col = plot_idx // n_cols, plot_idx % n_cols
            axes[row, col].plot(self.metrics_history[metric], label=metric)
            axes[row, col].set_title(f'{metric.replace("_", " ").title()}')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Loss')
            axes[row, col].grid(True)
            axes[row, col].legend()
            plot_idx += 1
        
        # 绘制学习率曲线
        if self.learning_rates:
            row, col = plot_idx // n_cols, plot_idx % n_cols
            axes[row, col].plot(self.learning_rates, label='Learning Rate')
            axes[row, col].set_title('Learning Rate Schedule')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Learning Rate')
            axes[row, col].set_yscale('log')
            axes[row, col].grid(True)
            axes[row, col].legend()
            plot_idx += 1
        
        # 绘制梯度范数曲线
        if self.gradient_norms:
            row, col = plot_idx // n_cols, plot_idx % n_cols
            axes[row, col].plot(self.gradient_norms, label='Gradient Norm')
            axes[row, col].set_title('Gradient Norm')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel('Gradient Norm')
            axes[row, col].grid(True)
            axes[row, col].legend()
            plot_idx += 1
        
        # 隐藏未使用的子图
        for i in range(plot_idx, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.save_dir, 'training_curves.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"训练曲线已保存到: {save_path}")
    
    def check_convergence(self, patience: int = 10) -> Dict[str, Any]:
        """检查收敛状态"""
        if 'val_loss' not in self.metrics_history or len(self.metrics_history['val_loss']) < patience:
            return {'converged': False, 'reason': '数据不足'}
        
        recent_losses = self.metrics_history['val_loss'][-patience:]
        
        # 检查是否还在改善
        best_recent = min(recent_losses)
        overall_best = min(self.metrics_history['val_loss'])
        
        # 检查梯度趋势
        if len(recent_losses) >= 5:
            # 计算最近几个epoch的斜率
            x = np.arange(len(recent_losses))
            slope = np.polyfit(x, recent_losses, 1)[0]
            
            convergence_info = {
                'converged': abs(slope) < 1e-6,  # 斜率接近0
                'slope': slope,
                'recent_improvement': best_recent < overall_best * 1.01,  # 1%容差
                'stagnant_epochs': patience if best_recent >= overall_best * 1.01 else 0
            }
        else:
            convergence_info = {
                'converged': False,
                'reason': '需要更多epoch来判断收敛'
            }
        
        return convergence_info

def enhanced_pretrain_forward_model(use_small_dataset_strategy: bool = True,
                                  enable_validation: bool = True,
                                  save_plots: bool = True) -> Dict[str, Any]:
    """增强的前向模型预训练"""
    logger.info("开始增强的前向模型预训练...")
    
    # 创建必要的目录
    os.makedirs(config.SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    
    # 获取数据加载器
    train_loader, val_loader, _, scalers = get_dataloaders(batch_size=config.BATCH_SIZE)
    
    # 初始化模型
    model = ForwardModel(
        input_dim=len(config.STRUCT_PARAMS),
        output_dim=len(config.SPECTRA_PARAMS)
    ).to(config.DEVICE)
    
    # 选择训练策略
    if use_small_dataset_strategy:
        logger.info("使用小数据集专用训练策略")
        trainer = SmallDatasetTrainer(model, config.DEVICE)
        trainer.setup_optimizer_and_scheduler(config.PRETRAIN_FWD_MODEL_LR)
        
        # 使用增强的训练器
        training_results = trainer.train(
            train_loader, 
            val_loader, 
            epochs=config.PRETRAIN_FWD_MODEL_EPOCHS,
            save_path=os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth')
        )
        
        # 绘制训练历史
        if save_plots:
            trainer.plot_training_history(
                os.path.join(config.PLOT_DIR, 'enhanced_forward_model_training.png')
            )
    
    else:
        logger.info("使用标准训练策略")
        # 标准训练流程（保持原有逻辑）
        training_results = _standard_training(model, train_loader, val_loader, scalers)
    
    # 前向模型质量验证
    validation_results = None
    if enable_validation:
        validator = ForwardModelValidator()
        validation_results = validator.validate_model_quality(model, val_loader, scalers)
        
        if not validation_results['quality_check']['overall_passed']:
            logger.warning("前向模型质量验证未通过！建议调整训练参数或增加训练时间")
            
            # 提供改进建议
            suggestions = _generate_improvement_suggestions(validation_results)
            logger.info("改进建议:")
            for suggestion in suggestions:
                logger.info(f"  - {suggestion}")
    
    # 汇总结果
    results = {
        'training_results': training_results,
        'validation_results': validation_results,
        'model_path': os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth'),
        'scalers': scalers
    }
    
    logger.info("增强的前向模型预训练完成")
    
    return results

def _standard_training(model, train_loader, val_loader, scalers):
    """标准训练流程（保持兼容性）"""
    # 这里保持原有的训练逻辑，作为fallback
    optimizer = optim.Adam(model.parameters(), lr=config.PRETRAIN_FWD_MODEL_LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    
    # 训练监控
    monitor = TrainingMonitor()
    
    # 损失函数设置
    spectra_criterion = nn.MSELoss()
    metrics_scale = torch.tensor(scalers['metrics'].scale_, dtype=torch.float32).to(config.DEVICE)
    metrics_offset = torch.tensor(scalers['metrics'].min_, dtype=torch.float32).to(config.DEVICE)
    metric_loss_weights = 1.0 / (metrics_scale + 1e-8)
    metrics_criterion = WeightedMSELoss(weights=metric_loss_weights)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 30  # 增加patience
    
    for epoch in range(config.PRETRAIN_FWD_MODEL_EPOCHS):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_spectra_loss = 0.0
        train_metrics_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.PRETRAIN_FWD_MODEL_EPOCHS}")
        for batch in progress_bar:
            struct_params = batch['struct'].to(config.DEVICE)
            target_spectra = batch['spectra'].to(config.DEVICE)
            target_metrics = batch['metrics'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            predicted_spectra, predicted_metrics = model(struct_params)
            
            # 计算损失
            loss_spectra = spectra_criterion(predicted_spectra, target_spectra)
            
            predicted_metrics_cleaned = torch.nan_to_num(predicted_metrics, nan=0.0)
            predicted_metrics_scaled = predicted_metrics_cleaned * metrics_scale + metrics_offset
            target_metrics_cleaned = torch.nan_to_num(target_metrics, nan=0.0)
            loss_metrics = metrics_criterion(predicted_metrics_scaled, target_metrics_cleaned)
            
            loss = config.SPECTRA_LOSS_WEIGHT * loss_spectra + config.METRIC_LOSS_WEIGHT * loss_metrics
            
            loss.backward()
            
            # 梯度裁剪和记录
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_spectra_loss += loss_spectra.item()
            train_metrics_loss += loss_metrics.item()
            
            progress_bar.set_postfix({
                'loss': loss.item(),
                'grad_norm': grad_norm.item()
            })
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_spectra_loss = 0.0
        val_metrics_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                struct_params = batch['struct'].to(config.DEVICE)
                target_spectra = batch['spectra'].to(config.DEVICE)
                target_metrics = batch['metrics'].to(config.DEVICE)
                
                predicted_spectra, predicted_metrics = model(struct_params)
                
                loss_spectra = spectra_criterion(predicted_spectra, target_spectra)
                predicted_metrics_cleaned = torch.nan_to_num(predicted_metrics, nan=0.0)
                predicted_metrics_scaled = predicted_metrics_cleaned * metrics_scale + metrics_offset
                target_metrics_cleaned = torch.nan_to_num(target_metrics, nan=0.0)
                loss_metrics = metrics_criterion(predicted_metrics_scaled, target_metrics_cleaned)
                
                loss = config.SPECTRA_LOSS_WEIGHT * loss_spectra + config.METRIC_LOSS_WEIGHT * loss_metrics
                
                val_loss += loss.item()
                val_spectra_loss += loss_spectra.item()
                val_metrics_loss += loss_metrics.item()
        
        # 计算平均损失
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录指标
        epoch_metrics = {
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_spectra_loss': train_spectra_loss / len(train_loader),
            'val_spectra_loss': val_spectra_loss / len(val_loader),
            'train_metrics_loss': train_metrics_loss / len(train_loader),
            'val_metrics_loss': val_metrics_loss / len(val_loader)
        }
        
        monitor.log_epoch_metrics(epoch, epoch_metrics, current_lr, grad_norm.item())
        
        # 学习率调度和早停
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(config.SAVED_MODELS_DIR, 'best_forward_model.pth'))
        else:
            patience_counter += 1
        
        logger.info(f"Epoch {epoch+1}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, LR={current_lr:.2e}")
        
        if patience_counter >= patience:
            logger.info(f"早停触发，在第 {epoch+1} epoch停止")
            break
    
    # 保存训练曲线
    monitor.plot_training_curves()
    
    return {
        'best_val_loss': best_val_loss,
        'total_epochs': epoch + 1,
        'metrics_history': dict(monitor.metrics_history)
    }

def _generate_improvement_suggestions(validation_results: Dict[str, Any]) -> List[str]:
    """根据验证结果生成改进建议"""
    suggestions = []
    
    quality_check = validation_results['quality_check']
    results = validation_results
    
    if not quality_check['spectra_mse_passed']:
        suggestions.append(f"光谱MSE过高 ({results['spectra_mse']:.6f})，考虑增加训练轮数或调整网络结构")
    
    if not quality_check['metrics_mse_passed']:
        suggestions.append(f"指标MSE过高 ({results['metrics_mse']:.6f})，考虑调整指标损失权重或改进特征提取器")
    
    if results['spectra_r2'] < 0.8:
        suggestions.append("光谱R²得分较低，考虑增加模型复杂度或使用正则化技术")
    
    if results['metrics_r2'] < 0.8:
        suggestions.append("指标R²得分较低，考虑改进指标提取的物理模型")
    
    # 检查各个指标的表现
    per_metric = results.get('per_metric_analysis', {})
    poor_metrics = [name for name, metrics in per_metric.items() if metrics['r2'] < 0.7]
    
    if poor_metrics:
        suggestions.append(f"以下指标预测较差: {', '.join(poor_metrics)}，考虑针对性优化")
    
    return suggestions

if __name__ == '__main__':
    # 执行增强的前向模型预训练
    results = enhanced_pretrain_forward_model(
        use_small_dataset_strategy=True,
        enable_validation=True,
        save_plots=True
    )
    
    print("=== 训练结果摘要 ===")
    if results['validation_results']:
        quality = results['validation_results']['quality_check']
        print(f"模型质量验证: {'通过' if quality['overall_passed'] else '未通过'}")
        print(f"通过检查: {quality['passed_count']}/{quality['total_checks']}")
    
    print("增强训练流程测试完成！")
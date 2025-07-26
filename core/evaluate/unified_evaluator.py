# PI_GAN_THZ/core/evaluate/unified_evaluator.py

import sys
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import time
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入增强模型
from core.models.enhanced_generator import EnhancedGenerator
from core.models.enhanced_discriminator import EnhancedDiscriminator
from core.models.enhanced_forward_model import EnhancedForwardPINN

import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics, normalize_spectrum
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_mse, criterion_bce
from core.utils.visualization import EvaluationVisualizer

class UnifiedEvaluator:
    """
    统一评估器：包含前向网络评估、PI-GAN评估、结构预测和模型验证
    """
    
    def __init__(self, device: str = "auto"):
        """
        初始化统一评估器
        
        Args:
            device: 计算设备 ("auto", "cpu", "cuda")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.generator = None
        self.discriminator = None
        self.forward_model = None
        self.evaluation_results = {}
        
        # 初始化可视化器
        plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
        self.visualizer = EvaluationVisualizer(save_dir=plots_dir)
        
        print(f"Unified Evaluator initialized on device: {self.device}")
    
    def load_models(self, model_dir: str = None) -> bool:
        """
        加载训练好的增强模型
        
        Args:
            model_dir: 模型保存目录
            
        Returns:
            bool: 加载是否成功
        """
        return self.load_models_enhanced(model_dir)
    
    def load_models_enhanced(self, model_dir: str = None) -> bool:
        """
        加载增强版训练好的模型
        
        Args:
            model_dir: 模型保存目录
            
        Returns:
            bool: 加载是否成功
        """
        if model_dir is None:
            model_dir = cfg.SAVED_MODELS_DIR
            
        print(f"Loading enhanced models from: {model_dir}")
        
        try:
            # 初始化增强模型
            self.generator = EnhancedGenerator(
                spectrum_dim=cfg.SPECTRUM_DIM,
                z_dim=cfg.Z_DIM,
                output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM
            ).to(self.device)
            
            self.discriminator = EnhancedDiscriminator(
                spectrum_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM,
                param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM
            ).to(self.device)
            
            self.forward_model = EnhancedForwardPINN(
                input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM
            ).to(self.device)
            
            # 加载权重
            gen_path = os.path.join(model_dir, "generator_enhanced_final.pth")
            disc_path = os.path.join(model_dir, "discriminator_enhanced_final.pth")
            fwd_path = os.path.join(model_dir, "forward_model_enhanced_final.pth")
            
            # 如果增强模型不存在，尝试加载原始模型
            if not all(os.path.exists(p) for p in [gen_path, disc_path, fwd_path]):
                print("Enhanced model files not found, trying original models...")
                gen_path = os.path.join(model_dir, "generator_final.pth")
                disc_path = os.path.join(model_dir, "discriminator_final.pth")
                fwd_path = os.path.join(model_dir, "forward_model_final.pth")
                
            if not all(os.path.exists(p) for p in [gen_path, disc_path, fwd_path]):
                print("Error: Model files not found!")
                return False
                
            self.generator.load_state_dict(torch.load(gen_path, map_location=self.device))
            self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))
            self.forward_model.load_state_dict(torch.load(fwd_path, map_location=self.device))
            
            # 设置为评估模式
            self.generator.eval()
            self.discriminator.eval()
            self.forward_model.eval()
            
            print("✓ Enhanced models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error loading enhanced models: {e}")
            return False
    
    def load_dataset(self, data_path: str = None) -> bool:
        """
        加载数据集
        
        Args:
            data_path: 数据集路径
            
        Returns:
            bool: 加载是否成功
        """
        if data_path is None:
            data_path = cfg.DATASET_PATH
            
        try:
            self.dataset = MetamaterialDataset(
                data_path=data_path, 
                num_points_per_sample=cfg.SPECTRUM_DIM
            )
            print(f"✓ Dataset loaded: {len(self.dataset)} samples")
            return True
            
        except Exception as e:
            print(f"✗ Error loading dataset: {e}")
            return False
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算回归评估指标
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            Dict[str, float]: 评估指标字典
        """
        metrics = {}
        
        # 基础回归指标
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # 相关性指标
        try:
            metrics['r2'] = r2_score(y_true, y_pred)
        except:
            metrics['r2'] = float('nan')
            
        # 相关系数
        try:
            if y_true.ndim == 1:
                pearson_corr, _ = pearsonr(y_true, y_pred)
                metrics['pearson_r'] = pearson_corr
            else:
                # 多维数据计算平均相关系数
                pearson_corrs = []
                for i in range(y_true.shape[1]):
                    try:
                        p_corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
                        pearson_corrs.append(p_corr)
                    except:
                        pass
                        
                metrics['pearson_r'] = np.mean(pearson_corrs) if pearson_corrs else float('nan')
        except:
            metrics['pearson_r'] = float('nan')
        
        # 相对误差
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return metrics
    
    def evaluate_forward_network(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        评估前向网络性能
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 前向网络评估结果
        """
        print(f"\n=== Forward Network Evaluation ({num_samples} samples) ===")
        
        if self.forward_model is None or self.dataset is None:
            raise ValueError("Forward model and dataset must be loaded first!")
        
        # 随机采样
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False)
        
        all_real_spectra = []
        all_pred_spectra = []
        all_real_metrics = []
        all_pred_metrics = []
        
        with torch.no_grad():
            for batch in dataloader:
                real_spectrum, _, real_params_norm, real_metrics_denorm, real_metrics_norm = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                real_metrics_norm = real_metrics_norm.to(self.device)
                
                # 前向模型预测
                if isinstance(self.forward_model, EnhancedForwardPINN):
                    pred_spectrum, _ = self.forward_model(real_params_norm)
                    pred_metrics_norm = torch.zeros_like(real_metrics_norm)  # 简化处理
                else:
                    pred_spectrum, pred_metrics_norm = self.forward_model(real_params_norm)
                pred_metrics_denorm = denormalize_metrics(pred_metrics_norm, self.dataset.metric_ranges)
                
                # 收集结果
                all_real_spectra.append(real_spectrum.cpu().numpy())
                all_pred_spectra.append(pred_spectrum.cpu().numpy())
                all_real_metrics.append(real_metrics_denorm.cpu().numpy())
                all_pred_metrics.append(pred_metrics_denorm.cpu().numpy())
        
        # 合并结果
        all_real_spectra = np.concatenate(all_real_spectra, axis=0)
        all_pred_spectra = np.concatenate(all_pred_spectra, axis=0)
        all_real_metrics = np.concatenate(all_real_metrics, axis=0)
        all_pred_metrics = np.concatenate(all_pred_metrics, axis=0)
        
        # 计算评估指标
        spectrum_metrics = self.calculate_metrics(all_real_spectra, all_pred_spectra)
        metrics_metrics = self.calculate_metrics(all_real_metrics, all_pred_metrics)
        
        results = {
            'spectrum_prediction': spectrum_metrics,
            'metrics_prediction': metrics_metrics,
            'num_samples': len(all_real_spectra),
            'data_samples': {
                'real_spectra': all_real_spectra[:50],  # 保存前50个样本用于可视化
                'pred_spectra': all_pred_spectra[:50],
                'real_metrics': all_real_metrics[:50],
                'pred_metrics': all_pred_metrics[:50]
            }
        }
        
        print(f"✓ Forward network evaluation completed")
        print(f"  - Spectrum R²: {spectrum_metrics['r2']:.4f}")
        print(f"  - Metrics R²: {metrics_metrics['r2']:.4f}")
        
        return results
    
    def evaluate_pigan(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        评估PI-GAN性能（生成器和判别器）
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: PI-GAN评估结果
        """
        print(f"\n=== PI-GAN Evaluation ({num_samples} samples) ===")
        
        if not all([self.generator, self.discriminator, self.forward_model, self.dataset]):
            raise ValueError("All models and dataset must be loaded first!")
        
        # 随机采样
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False)
        
        # 生成器评估
        all_real_params = []
        all_pred_params = []
        all_real_scores = []
        all_fake_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_denorm = real_params_denorm.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                # 生成器预测参数
                if isinstance(self.generator, EnhancedGenerator):
                    latent_vector = torch.randn(real_spectrum.shape[0], cfg.Z_DIM).to(self.device)
                    pred_params_norm = self.generator(real_spectrum, latent_vector)
                else:
                    pred_params_norm = self.generator(real_spectrum)
                pred_params_denorm = denormalize_params(pred_params_norm, self.dataset.param_ranges)
                
                # 判别器评分
                if isinstance(self.discriminator, EnhancedDiscriminator):
                    real_scores, _ = self.discriminator(real_spectrum, real_params_denorm)
                    fake_scores, _ = self.discriminator(real_spectrum, pred_params_denorm)
                else:
                    real_scores = self.discriminator(real_spectrum, real_params_denorm)
                    fake_scores = self.discriminator(real_spectrum, pred_params_denorm)
                
                # 收集结果
                all_real_params.append(real_params_denorm.cpu().numpy())
                all_pred_params.append(pred_params_denorm.cpu().numpy())
                all_real_scores.append(real_scores.cpu().numpy())
                all_fake_scores.append(fake_scores.cpu().numpy())
        
        # 合并结果
        all_real_params = np.concatenate(all_real_params, axis=0)
        all_pred_params = np.concatenate(all_pred_params, axis=0)
        all_real_scores = np.concatenate(all_real_scores, axis=0)
        all_fake_scores = np.concatenate(all_fake_scores, axis=0)
        
        # 计算评估指标
        param_metrics = self.calculate_metrics(all_real_params, all_pred_params)
        
        # 判别器性能
        real_accuracy = np.mean(all_real_scores > 0.5)
        fake_accuracy = np.mean(all_fake_scores < 0.5)
        overall_accuracy = (real_accuracy + fake_accuracy) / 2
        
        results = {
            'parameter_prediction': param_metrics,
            'discriminator_performance': {
                'real_accuracy': real_accuracy,
                'fake_accuracy': fake_accuracy,
                'overall_accuracy': overall_accuracy,
                'real_score_mean': np.mean(all_real_scores),
                'fake_score_mean': np.mean(all_fake_scores)
            },
            'num_samples': len(all_real_params),
            'data_samples': {
                'real_params': all_real_params[:50],  # 保存前50个样本用于可视化
                'pred_params': all_pred_params[:50]
            },
            'score_distributions': {
                'real_scores': all_real_scores[:200],  # 保存前200个得分用于可视化
                'fake_scores': all_fake_scores[:200]
            }
        }
        
        print(f"✓ PI-GAN evaluation completed")
        print(f"  - Parameter R²: {param_metrics['r2']:.4f}")
        print(f"  - Discriminator Accuracy: {overall_accuracy:.4f}")
        
        return results
    
    def evaluate_pigan_enhanced(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        评估增强版PI-GAN性能（生成器、判别器和物理一致性）
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 增强版PI-GAN评估结果
        """
        print(f"\n=== Enhanced PI-GAN Evaluation ({num_samples} samples) ===")
        
        if not all([self.generator, self.discriminator, self.forward_model, self.dataset]):
            raise ValueError("All models and dataset must be loaded first!")
        
        # 确认使用的是增强模型
        if not (isinstance(self.generator, EnhancedGenerator) and 
                isinstance(self.discriminator, EnhancedDiscriminator)):
            print("Warning: Models are not enhanced versions. Using standard evaluation.")
            return self.evaluate_pigan(num_samples)
        
        # 随机采样
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False)
        
        # 生成器评估
        all_real_params = []
        all_pred_params = []
        all_real_scores = []
        all_fake_scores = []
        all_physics_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_denorm = real_params_denorm.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                # 生成器预测参数
                latent_vector = torch.randn(real_spectrum.shape[0], cfg.Z_DIM).to(self.device)
                pred_params_norm = self.generator(real_spectrum, latent_vector)
                pred_params_denorm = denormalize_params(pred_params_norm, self.dataset.param_ranges)
                
                # 判别器评分
                real_scores_rf, real_scores_physics = self.discriminator(real_spectrum, real_params_denorm)
                fake_scores_rf, fake_scores_physics = self.discriminator(real_spectrum, pred_params_denorm)
                
                # 收集结果
                all_real_params.append(real_params_denorm.cpu().numpy())
                all_pred_params.append(pred_params_denorm.cpu().numpy())
                all_real_scores.append(real_scores_rf.cpu().numpy())
                all_fake_scores.append(fake_scores_rf.cpu().numpy())
                all_physics_scores.append(fake_scores_physics.cpu().numpy())
        
        # 合并结果
        all_real_params = np.concatenate(all_real_params, axis=0)
        all_pred_params = np.concatenate(all_pred_params, axis=0)
        all_real_scores = np.concatenate(all_real_scores, axis=0)
        all_fake_scores = np.concatenate(all_fake_scores, axis=0)
        all_physics_scores = np.concatenate(all_physics_scores, axis=0)
        
        # 计算评估指标
        param_metrics = self.calculate_metrics(all_real_params, all_pred_params)
        
        # 判别器性能
        real_accuracy = np.mean(all_real_scores > 0.5)
        fake_accuracy = np.mean(all_fake_scores < 0.5)
        overall_accuracy = (real_accuracy + fake_accuracy) / 2
        
        # 物理一致性性能
        physics_score_mean = np.mean(all_physics_scores)
        physics_score_std = np.std(all_physics_scores)
        
        results = {
            'parameter_prediction': param_metrics,
            'discriminator_performance': {
                'real_accuracy': real_accuracy,
                'fake_accuracy': fake_accuracy,
                'overall_accuracy': overall_accuracy,
                'real_score_mean': np.mean(all_real_scores),
                'fake_score_mean': np.mean(all_fake_scores)
            },
            'physics_consistency': {
                'physics_score_mean': physics_score_mean,
                'physics_score_std': physics_score_std
            },
            'num_samples': len(all_real_params),
            'data_samples': {
                'real_params': all_real_params[:50],  # 保存前50个样本用于可视化
                'pred_params': all_pred_params[:50]
            },
            'score_distributions': {
                'real_scores': all_real_scores[:200],  # 保存前200个得分用于可视化
                'fake_scores': all_fake_scores[:200],
                'physics_scores': all_physics_scores[:200]
            }
        }
        
        print(f"✓ Enhanced PI-GAN evaluation completed")
        print(f"  - Parameter R²: {param_metrics['r2']:.4f}")
        print(f"  - Discriminator Accuracy: {overall_accuracy:.4f}")
        print(f"  - Physics Consistency Score: {physics_score_mean:.4f}")
        
        return results
    
    def evaluate_structural_prediction(self, num_samples: int = 100) -> Dict[str, Any]:
        """
        评估结构预测能力
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 结构预测评估结果
        """
        print(f"\n=== Structural Prediction Evaluation ({num_samples} samples) ===")
        
        if not all([self.generator, self.forward_model, self.dataset]):
            raise ValueError("Generator, forward model and dataset must be loaded first!")
        
        # 随机采样
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=32, shuffle=False)
        
        param_range_violations = []
        reconstruction_errors = []
        consistency_scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                # 生成器预测参数
                if isinstance(self.generator, EnhancedGenerator):
                    latent_vector = torch.randn(real_spectrum.shape[0], cfg.Z_DIM).to(self.device)
                    pred_params_norm = self.generator(real_spectrum, latent_vector)
                else:
                    pred_params_norm = self.generator(real_spectrum)
                
                # 检查参数范围约束
                range_violations = torch.sum((pred_params_norm < 0) | (pred_params_norm > 1), dim=1).cpu().numpy()
                param_range_violations.extend(range_violations)
                
                # 前向模型重建光谱
                if isinstance(self.forward_model, EnhancedForwardPINN):
                    recon_spectrum, _ = self.forward_model(pred_params_norm)
                else:
                    recon_spectrum, _ = self.forward_model(pred_params_norm)
                
                # 重建误差
                recon_error = torch.mean((real_spectrum - recon_spectrum) ** 2, dim=1).cpu().numpy()
                reconstruction_errors.extend(recon_error)
                
                # 一致性得分 (1 - 归一化重建误差)
                consistency = 1.0 / (1.0 + recon_error)
                consistency_scores.extend(consistency)
        
        # 统计结果
        param_range_violations = np.array(param_range_violations)
        reconstruction_errors = np.array(reconstruction_errors)
        consistency_scores = np.array(consistency_scores)
        
        results = {
            'param_range_violation_rate': np.mean(param_range_violations > 0),
            'avg_param_violations': np.mean(param_range_violations),
            'reconstruction_error_mean': np.mean(reconstruction_errors),
            'reconstruction_error_std': np.std(reconstruction_errors),
            'consistency_score_mean': np.mean(consistency_scores),
            'consistency_score_std': np.std(consistency_scores),
            'num_samples': len(param_range_violations)
        }
        
        print(f"✓ Structural prediction evaluation completed")
        print(f"  - Parameter violation rate: {results['param_range_violation_rate']:.4f}")
        print(f"  - Consistency score: {results['consistency_score_mean']:.4f}")
        
        return results
    
    def evaluate_model_validation(self, num_samples: int = 500) -> Dict[str, Any]:
        """
        模型验证评估
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 模型验证结果
        """
        print(f"\n=== Model Validation ({num_samples} samples) ===")
        
        if not all([self.generator, self.forward_model, self.dataset]):
            raise ValueError("Generator, forward model and dataset must be loaded first!")
        
        # 随机采样
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False)
        
        cycle_consistency_errors = []
        prediction_stability = []
        physical_plausibility = []
        
        with torch.no_grad():
            for batch in dataloader:
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                # 循环一致性测试: spectrum -> params -> spectrum
                if isinstance(self.generator, EnhancedGenerator):
                    latent_vector = torch.randn(real_spectrum.shape[0], cfg.Z_DIM).to(self.device)
                    pred_params_norm = self.generator(real_spectrum, latent_vector)
                else:
                    pred_params_norm = self.generator(real_spectrum)
                
                if isinstance(self.forward_model, EnhancedForwardPINN):
                    recon_spectrum, _ = self.forward_model(pred_params_norm)
                else:
                    recon_spectrum, _ = self.forward_model(pred_params_norm)
                
                cycle_error = torch.mean((real_spectrum - recon_spectrum) ** 2, dim=1).cpu().numpy()
                cycle_consistency_errors.extend(cycle_error)
                
                # 预测稳定性测试: 添加小噪声后的预测一致性
                noise = torch.randn_like(real_spectrum) * 0.01
                noisy_spectrum = real_spectrum + noise
                if isinstance(self.generator, EnhancedGenerator):
                    pred_params_noisy = self.generator(noisy_spectrum, latent_vector)
                else:
                    pred_params_noisy = self.generator(noisy_spectrum)
                
                stability = torch.mean((pred_params_norm - pred_params_noisy) ** 2, dim=1).cpu().numpy()
                prediction_stability.extend(stability)
                
                # 物理合理性: 预测参数的物理约束满足程度
                pred_params_denorm = denormalize_params(pred_params_norm, self.dataset.param_ranges)
                
                # 检查参数是否在合理范围内
                plausibility_score = torch.mean(
                    torch.sigmoid(pred_params_norm * 10 - 5), dim=1
                ).cpu().numpy()
                physical_plausibility.extend(plausibility_score)
        
        # 统计结果
        cycle_consistency_errors = np.array(cycle_consistency_errors)
        prediction_stability = np.array(prediction_stability)
        physical_plausibility = np.array(physical_plausibility)
        
        results = {
            'cycle_consistency_error_mean': np.mean(cycle_consistency_errors),
            'cycle_consistency_error_std': np.std(cycle_consistency_errors),
            'prediction_stability_mean': np.mean(prediction_stability),
            'prediction_stability_std': np.std(prediction_stability),
            'physical_plausibility_mean': np.mean(physical_plausibility),
            'physical_plausibility_std': np.std(physical_plausibility),
            'num_samples': len(cycle_consistency_errors)
        }
        
        print(f"✓ Model validation completed")
        print(f"  - Cycle consistency error: {results['cycle_consistency_error_mean']:.6f}")
        print(f"  - Prediction stability: {results['prediction_stability_mean']:.6f}")
        print(f"  - Physical plausibility: {results['physical_plausibility_mean']:.4f}")
        
        return results
    
    def run_comprehensive_evaluation(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        运行全面评估
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 完整评估结果
        """
        print("\n" + "="*80)
        print("PI-GAN COMPREHENSIVE EVALUATION")
        print("="*80)
        
        start_time = time.time()
        
        # 检查模型和数据集
        if not all([self.generator, self.discriminator, self.forward_model, self.dataset]):
            raise ValueError("All models and dataset must be loaded first!")
        
        # 执行各项评估
        results = {
            'forward_network_evaluation': self.evaluate_forward_network(num_samples),
            'pigan_evaluation': self.evaluate_pigan(num_samples),
            'structural_prediction_evaluation': self.evaluate_structural_prediction(min(num_samples//2, 500)),
            'model_validation': self.evaluate_model_validation(min(num_samples//2, 500)),
            'evaluation_time': time.time() - start_time,
            'total_samples': num_samples
        }
        
        # 保存结果
        self.evaluation_results = results
        
        # 生成可视化
        print(f"\n🎨 Generating evaluation visualizations...")
        self.generate_visualizations(results)
        
        print(f"\n" + "="*80)
        print(f"EVALUATION COMPLETED in {results['evaluation_time']:.2f}s")
        print("="*80)
        
        return results
    
    def run_comprehensive_evaluation_enhanced(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        运行增强版全面评估
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 完整评估结果
        """
        print("\n" + "="*80)
        print("ENHANCED PI-GAN COMPREHENSIVE EVALUATION")
        print("="*80)
        
        start_time = time.time()
        
        # 检查模型和数据集
        if not all([self.generator, self.discriminator, self.forward_model, self.dataset]):
            raise ValueError("All models and dataset must be loaded first!")
        
        # 执行各项评估
        pigan_eval = self.evaluate_pigan_enhanced(num_samples) if isinstance(self.discriminator, EnhancedDiscriminator) else self.evaluate_pigan(num_samples)
        
        results = {
            'forward_network_evaluation': self.evaluate_forward_network(num_samples),
            'pigan_evaluation': pigan_eval,
            'structural_prediction_evaluation': self.evaluate_structural_prediction(min(num_samples//2, 500)),
            'model_validation': self.evaluate_model_validation(min(num_samples//2, 500)),
            'evaluation_time': time.time() - start_time,
            'total_samples': num_samples
        }
        
        # 保存结果
        self.evaluation_results = results
        
        # 生成可视化
        print(f"\n🎨 Generating evaluation visualizations...")
        self.generate_visualizations(results)
        
        print(f"\n" + "="*80)
        print(f"ENHANCED EVALUATION COMPLETED in {results['evaluation_time']:.2f}s")
        print("="*80)
        
        return results
    
    def generate_visualizations(self, results: Dict[str, Any]) -> None:
        """
        生成所有评估结果的可视化
        
        Args:
            results: 完整评估结果
        """
        try:
            # 1. 前向网络评估可视化
            fwd_data = results['forward_network_evaluation'].get('data_samples', {})
            fwd_plot_path = self.visualizer.plot_forward_network_evaluation(
                results['forward_network_evaluation'], 
                fwd_data
            )
            print(f"✓ Forward network evaluation plot saved: {fwd_plot_path}")
            
            # 2. PI-GAN评估可视化
            pigan_data = results['pigan_evaluation'].get('data_samples', {})
            score_data = results['pigan_evaluation'].get('score_distributions', {})
            pigan_plot_path = self.visualizer.plot_pigan_evaluation(
                results['pigan_evaluation'],
                pigan_data,
                score_data
            )
            print(f"✓ PI-GAN evaluation plot saved: {pigan_plot_path}")
            
            # 3. 结构预测评估可视化
            struct_plot_path = self.visualizer.plot_structural_prediction_evaluation(
                results['structural_prediction_evaluation']
            )
            print(f"✓ Structural prediction evaluation plot saved: {struct_plot_path}")
            
            # 4. 模型验证评估可视化
            validation_plot_path = self.visualizer.plot_model_validation_evaluation(
                results['model_validation']
            )
            print(f"✓ Model validation evaluation plot saved: {validation_plot_path}")
            
            # 5. 综合摘要可视化
            summary_plot_path = self.visualizer.plot_comprehensive_summary(results)
            print(f"✓ Comprehensive summary plot saved: {summary_plot_path}")
            
            print(f"🎯 All evaluation visualizations generated in: {self.visualizer.save_dir}")
            
        except Exception as e:
            print(f"⚠ Warning: Failed to generate some visualizations: {e}")
    
    def generate_summary_report(self, save_path: str = None) -> str:
        """
        生成评估总结报告
        
        Args:
            save_path: 报告保存路径
            
        Returns:
            str: 报告内容
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run comprehensive evaluation first.")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("PI-GAN UNIFIED EVALUATION REPORT")
        report_lines.append("="*80)
        
        # 基本信息
        report_lines.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total Samples: {self.evaluation_results['total_samples']}")
        report_lines.append(f"Evaluation Time: {self.evaluation_results['evaluation_time']:.2f}s")
        report_lines.append("")
        
        # 1. 前向网络评估
        fwd_results = self.evaluation_results['forward_network_evaluation']
        report_lines.append("1. FORWARD NETWORK EVALUATION")
        report_lines.append("-" * 40)
        spectrum_r2 = fwd_results['spectrum_prediction']['r2']
        metrics_r2 = fwd_results['metrics_prediction']['r2']
        report_lines.append(f"Spectrum Prediction R²: {spectrum_r2:.4f}")
        report_lines.append(f"Metrics Prediction R²: {metrics_r2:.4f}")
        if spectrum_r2 > 0.9 and metrics_r2 > 0.9:
            report_lines.append("✓ Forward network shows EXCELLENT performance")
        elif spectrum_r2 > 0.8 and metrics_r2 > 0.8:
            report_lines.append("✓ Forward network shows GOOD performance")
        else:
            report_lines.append("⚠ Forward network needs improvement")
        report_lines.append("")
        
        # 2. PI-GAN评估
        pigan_results = self.evaluation_results['pigan_evaluation']
        report_lines.append("2. PI-GAN EVALUATION")
        report_lines.append("-" * 40)
        param_r2 = pigan_results['parameter_prediction']['r2']
        disc_acc = pigan_results['discriminator_performance']['overall_accuracy']
        report_lines.append(f"Parameter Prediction R²: {param_r2:.4f}")
        report_lines.append(f"Discriminator Accuracy: {disc_acc:.4f}")
        
        # 检查是否有物理一致性评估
        if 'physics_consistency' in pigan_results:
            physics_score = pigan_results['physics_consistency']['physics_score_mean']
            report_lines.append(f"Physics Consistency Score: {physics_score:.4f}")
        
        if param_r2 > 0.8 and disc_acc > 0.8:
            report_lines.append("✓ PI-GAN shows EXCELLENT performance")
        elif param_r2 > 0.6 and disc_acc > 0.7:
            report_lines.append("✓ PI-GAN shows GOOD performance")
        else:
            report_lines.append("⚠ PI-GAN needs improvement")
        report_lines.append("")
        
        # 3. 结构预测评估
        struct_results = self.evaluation_results['structural_prediction_evaluation']
        report_lines.append("3. STRUCTURAL PREDICTION EVALUATION")
        report_lines.append("-" * 40)
        violation_rate = struct_results['param_range_violation_rate']
        consistency = struct_results['consistency_score_mean']
        report_lines.append(f"Parameter Violation Rate: {violation_rate:.4f}")
        report_lines.append(f"Consistency Score: {consistency:.4f}")
        if violation_rate < 0.1 and consistency > 0.8:
            report_lines.append("✓ Structural prediction is RELIABLE")
        elif violation_rate < 0.2 and consistency > 0.6:
            report_lines.append("✓ Structural prediction is ACCEPTABLE")
        else:
            report_lines.append("⚠ Structural prediction needs improvement")
        report_lines.append("")
        
        # 4. 模型验证
        valid_results = self.evaluation_results['model_validation']
        report_lines.append("4. MODEL VALIDATION")
        report_lines.append("-" * 40)
        cycle_error = valid_results['cycle_consistency_error_mean']
        stability = valid_results['prediction_stability_mean']
        plausibility = valid_results['physical_plausibility_mean']
        report_lines.append(f"Cycle Consistency Error: {cycle_error:.6f}")
        report_lines.append(f"Prediction Stability: {stability:.6f}")
        report_lines.append(f"Physical Plausibility: {plausibility:.4f}")
        if cycle_error < 0.01 and stability < 0.01 and plausibility > 0.8:
            report_lines.append("✓ Model validation is EXCELLENT")
        elif cycle_error < 0.05 and stability < 0.05 and plausibility > 0.6:
            report_lines.append("✓ Model validation is GOOD")
        else:
            report_lines.append("⚠ Model validation shows concerns")
        report_lines.append("")
        
        # 总结
        report_lines.append("5. OVERALL ASSESSMENT")
        report_lines.append("-" * 40)
        excellent_count = sum([
            spectrum_r2 > 0.9 and metrics_r2 > 0.9,
            param_r2 > 0.8 and disc_acc > 0.8,
            violation_rate < 0.1 and consistency > 0.8,
            cycle_error < 0.01 and stability < 0.01 and plausibility > 0.8
        ])
        
        if excellent_count >= 3:
            report_lines.append("🎯 OVERALL RATING: EXCELLENT")
        elif excellent_count >= 2:
            report_lines.append("✅ OVERALL RATING: GOOD")
        else:
            report_lines.append("⚠️ OVERALL RATING: NEEDS IMPROVEMENT")
        
        report_lines.append("="*80)
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if save_path is None:
            save_path = os.path.join(self.visualizer.save_dir, "unified_evaluation_report.txt")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nEvaluation report saved to: {save_path}")
        return report_content

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Run unified PI-GAN evaluation")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to dataset CSV file')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建评估器
    evaluator = UnifiedEvaluator(device=args.device)
    
    # 加载模型和数据
    if not evaluator.load_models(args.model_dir):
        print("Failed to load models!")
        return
    
    if not evaluator.load_dataset(args.data_path):
        print("Failed to load dataset!")
        return
    
    # 运行评估
    results = evaluator.run_comprehensive_evaluation_enhanced(args.num_samples)
    
    # 生成报告
    evaluator.generate_summary_report()
    
    print("\n✅ Unified evaluation completed successfully!")

if __name__ == "__main__":
    main()
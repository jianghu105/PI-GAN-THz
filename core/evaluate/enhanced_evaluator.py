# PI_GAN_THZ/core/evaluate/enhanced_evaluator.py

import sys
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics, normalize_spectrum
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_mse, criterion_bce

class EnhancedEvaluator:
    """
    增强型评估器：提供全面的PI-GAN模型评估功能
    """
    
    def __init__(self, device: str = "auto"):
        """
        初始化评估器
        
        Args:
            device: 计算设备 ("auto", "cpu", "cuda")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.generator = None
        self.discriminator = None
        self.forward_model = None
        self.dataset = None
        self.evaluation_results = {}
        
        print(f"Enhanced Evaluator initialized on device: {self.device}")
    
    def load_models(self, model_dir: str = None) -> bool:
        """
        加载训练好的模型
        
        Args:
            model_dir: 模型保存目录
            
        Returns:
            bool: 加载是否成功
        """
        if model_dir is None:
            model_dir = cfg.SAVED_MODELS_DIR
            
        print(f"Loading models from: {model_dir}")
        
        try:
            # 初始化模型
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
            
            # 加载权重
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
            
            print("Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"Error loading models: {e}")
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
            print(f"Dataset loaded: {len(self.dataset)} samples")
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
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
                spearman_corr, _ = spearmanr(y_true, y_pred)
                metrics['pearson_r'] = pearson_corr
                metrics['spearman_r'] = spearman_corr
            else:
                # 多维数据计算平均相关系数
                pearson_corrs = []
                spearman_corrs = []
                for i in range(y_true.shape[1]):
                    try:
                        p_corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
                        s_corr, _ = spearmanr(y_true[:, i], y_pred[:, i])
                        pearson_corrs.append(p_corr)
                        spearman_corrs.append(s_corr)
                    except:
                        pass
                        
                metrics['pearson_r'] = np.mean(pearson_corrs) if pearson_corrs else float('nan')
                metrics['spearman_r'] = np.mean(spearman_corrs) if spearman_corrs else float('nan')
        except:
            metrics['pearson_r'] = float('nan')
            metrics['spearman_r'] = float('nan')
        
        # 相对误差
        metrics['mape'] = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        
        return metrics
    
    def evaluate_generator(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        评估生成器性能
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        print(f"Evaluating Generator with {num_samples} samples...")
        
        if self.generator is None or self.dataset is None:
            raise ValueError("Models and dataset must be loaded first!")
        
        # 随机采样
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False)
        
        all_real_params = []
        all_pred_params = []
        all_real_spectra = []
        all_pred_spectra = []
        all_real_metrics = []
        all_pred_metrics = []
        
        with torch.no_grad():
            for batch in dataloader:
                real_spectrum, real_params_denorm, real_params_norm, real_metrics_denorm, real_metrics_norm = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                real_metrics_norm = real_metrics_norm.to(self.device)
                
                # 生成器预测
                pred_params_norm = self.generator(real_spectrum)
                pred_params_denorm = denormalize_params(pred_params_norm, self.dataset.param_ranges)
                
                # 前向模型验证
                pred_spectrum, pred_metrics_norm = self.forward_model(pred_params_norm)
                pred_metrics_denorm = denormalize_metrics(pred_metrics_norm, self.dataset.metric_ranges)
                
                # 收集结果
                all_real_params.append(real_params_denorm.cpu().numpy())
                all_pred_params.append(pred_params_denorm.cpu().numpy())
                all_real_spectra.append(real_spectrum.cpu().numpy())
                all_pred_spectra.append(pred_spectrum.cpu().numpy())
                all_real_metrics.append(real_metrics_denorm.cpu().numpy())
                all_pred_metrics.append(pred_metrics_denorm.cpu().numpy())
        
        # 合并结果
        all_real_params = np.concatenate(all_real_params, axis=0)
        all_pred_params = np.concatenate(all_pred_params, axis=0)
        all_real_spectra = np.concatenate(all_real_spectra, axis=0)
        all_pred_spectra = np.concatenate(all_pred_spectra, axis=0)
        all_real_metrics = np.concatenate(all_real_metrics, axis=0)
        all_pred_metrics = np.concatenate(all_pred_metrics, axis=0)
        
        # 计算评估指标
        results = {
            'parameter_prediction': self.calculate_regression_metrics(all_real_params, all_pred_params),
            'spectrum_reconstruction': self.calculate_regression_metrics(all_real_spectra, all_pred_spectra),
            'metrics_prediction': self.calculate_regression_metrics(all_real_metrics, all_pred_metrics),
            'num_samples': len(all_real_params),
            'data': {
                'real_params': all_real_params,
                'pred_params': all_pred_params,
                'real_spectra': all_real_spectra,
                'pred_spectra': all_pred_spectra,
                'real_metrics': all_real_metrics,
                'pred_metrics': all_pred_metrics
            }
        }
        
        print(f"Generator evaluation completed with {len(all_real_params)} samples")
        return results
    
    def evaluate_discriminator(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        评估判别器性能
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        print(f"Evaluating Discriminator with {num_samples} samples...")
        
        if self.discriminator is None or self.generator is None or self.dataset is None:
            raise ValueError("Models and dataset must be loaded first!")
        
        # 随机采样
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False)
        
        all_real_scores = []
        all_fake_scores = []
        bce_criterion = criterion_bce()
        
        with torch.no_grad():
            for batch in dataloader:
                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch
                
                real_spectrum = real_spectrum.to(self.device)
                real_params_denorm = real_params_denorm.to(self.device)
                real_params_norm = real_params_norm.to(self.device)
                
                batch_size = real_spectrum.size(0)
                
                # 真实样本评分
                real_scores = self.discriminator(real_spectrum, real_params_denorm)
                all_real_scores.append(real_scores.cpu().numpy())
                
                # 生成假样本
                fake_params_norm = self.generator(real_spectrum)
                fake_params_denorm = denormalize_params(fake_params_norm, self.dataset.param_ranges)
                
                # 假样本评分
                fake_scores = self.discriminator(real_spectrum, fake_params_denorm)
                all_fake_scores.append(fake_scores.cpu().numpy())
        
        # 合并结果
        all_real_scores = np.concatenate(all_real_scores, axis=0)
        all_fake_scores = np.concatenate(all_fake_scores, axis=0)
        
        # 计算判别器性能指标
        real_labels = np.ones_like(all_real_scores)
        fake_labels = np.zeros_like(all_fake_scores)
        
        # 准确率计算
        real_accuracy = np.mean(all_real_scores > 0.5)
        fake_accuracy = np.mean(all_fake_scores < 0.5)
        overall_accuracy = (real_accuracy + fake_accuracy) / 2
        
        # 损失计算
        real_loss = -np.mean(np.log(all_real_scores + 1e-8))
        fake_loss = -np.mean(np.log(1 - all_fake_scores + 1e-8))
        total_loss = real_loss + fake_loss
        
        results = {
            'real_accuracy': real_accuracy,
            'fake_accuracy': fake_accuracy,
            'overall_accuracy': overall_accuracy,
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'total_loss': total_loss,
            'real_score_mean': np.mean(all_real_scores),
            'fake_score_mean': np.mean(all_fake_scores),
            'real_score_std': np.std(all_real_scores),
            'fake_score_std': np.std(all_fake_scores),
            'num_samples': len(all_real_scores)
        }
        
        print(f"Discriminator evaluation completed with {len(all_real_scores)} samples")
        return results
    
    def evaluate_physics_consistency(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        评估物理一致性
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 物理一致性评估结果
        """
        print(f"Evaluating Physics Consistency with {num_samples} samples...")
        
        if self.generator is None or self.forward_model is None or self.dataset is None:
            raise ValueError("Models and dataset must be loaded first!")
        
        # 随机采样
        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)
        subset = Subset(self.dataset, sample_indices)
        dataloader = DataLoader(subset, batch_size=64, shuffle=False)
        
        param_range_violations = []
        spectrum_smoothness_scores = []
        frequency_responses = []
        
        with torch.no_grad():
            for batch in dataloader:
                real_spectrum, _, _, _, _ = batch
                real_spectrum = real_spectrum.to(self.device)
                
                # 生成器预测参数
                pred_params_norm = self.generator(real_spectrum)
                
                # 检查参数范围约束
                range_violations = torch.sum((pred_params_norm < 0) | (pred_params_norm > 1), dim=1).cpu().numpy()
                param_range_violations.extend(range_violations)
                
                # 前向模型预测
                pred_spectrum, pred_metrics = self.forward_model(pred_params_norm)
                
                # 光谱平滑性评估
                spectrum_diff = torch.diff(pred_spectrum, dim=1)
                smoothness = -torch.mean(spectrum_diff ** 2, dim=1).cpu().numpy()
                spectrum_smoothness_scores.extend(smoothness)
                
                # 频率响应分析
                freq_response = torch.mean(pred_spectrum, dim=1).cpu().numpy()
                frequency_responses.extend(freq_response)
        
        # 统计结果
        param_range_violations = np.array(param_range_violations)
        spectrum_smoothness_scores = np.array(spectrum_smoothness_scores)
        frequency_responses = np.array(frequency_responses)
        
        results = {
            'param_range_violation_rate': np.mean(param_range_violations > 0),
            'avg_param_violations_per_sample': np.mean(param_range_violations),
            'spectrum_smoothness_mean': np.mean(spectrum_smoothness_scores),
            'spectrum_smoothness_std': np.std(spectrum_smoothness_scores),
            'frequency_response_mean': np.mean(frequency_responses),
            'frequency_response_std': np.std(frequency_responses),
            'num_samples': len(param_range_violations)
        }
        
        print(f"Physics consistency evaluation completed with {len(param_range_violations)} samples")
        return results
    
    def comprehensive_evaluation(self, num_samples: int = 1000) -> Dict[str, Any]:
        """
        执行全面评估
        
        Args:
            num_samples: 评估样本数
            
        Returns:
            Dict[str, Any]: 完整评估结果
        """
        print("\n=== Starting Comprehensive Evaluation ===")
        start_time = time.time()
        
        # 检查模型和数据集
        if not all([self.generator, self.discriminator, self.forward_model, self.dataset]):
            raise ValueError("Models and dataset must be loaded first!")
        
        # 执行各项评估
        results = {
            'generator_evaluation': self.evaluate_generator(num_samples),
            'discriminator_evaluation': self.evaluate_discriminator(num_samples),
            'physics_consistency': self.evaluate_physics_consistency(num_samples),
            'evaluation_time': time.time() - start_time,
            'num_samples': num_samples
        }
        
        # 保存结果
        self.evaluation_results = results
        
        print(f"\n=== Comprehensive Evaluation Completed in {results['evaluation_time']:.2f}s ===")
        return results
    
    def save_evaluation_results(self, save_path: str = None) -> None:
        """
        保存评估结果
        
        Args:
            save_path: 保存路径
        """
        if save_path is None:
            save_path = os.path.join(cfg.SAVED_MODELS_DIR, "evaluation_results.npz")
        
        # 准备保存数据（移除不能序列化的数据）
        save_data = {}
        for key, value in self.evaluation_results.items():
            if key != 'generator_evaluation' or 'data' not in value:
                save_data[key] = value
            else:
                # 保存生成器评估结果但排除原始数据
                save_data[key] = {k: v for k, v in value.items() if k != 'data'}
        
        np.savez(save_path, **save_data)
        print(f"Evaluation results saved to: {save_path}")
    
    def generate_report(self, save_path: str = None) -> str:
        """
        生成评估报告
        
        Args:
            save_path: 报告保存路径
            
        Returns:
            str: 报告内容
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run comprehensive_evaluation first.")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("PI-GAN MODEL EVALUATION REPORT")
        report_lines.append("=" * 80)
        
        # 基本信息
        report_lines.append(f"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of Samples: {self.evaluation_results['num_samples']}")
        report_lines.append(f"Evaluation Time: {self.evaluation_results['evaluation_time']:.2f}s")
        report_lines.append("")
        
        # 生成器评估
        gen_results = self.evaluation_results['generator_evaluation']
        report_lines.append("1. GENERATOR EVALUATION")
        report_lines.append("-" * 40)
        
        report_lines.append("Parameter Prediction:")
        param_metrics = gen_results['parameter_prediction']
        report_lines.append(f"  - MSE: {param_metrics['mse']:.6f}")
        report_lines.append(f"  - MAE: {param_metrics['mae']:.6f}")
        report_lines.append(f"  - RMSE: {param_metrics['rmse']:.6f}")
        report_lines.append(f"  - R²: {param_metrics['r2']:.4f}")
        report_lines.append(f"  - Pearson R: {param_metrics['pearson_r']:.4f}")
        report_lines.append(f"  - MAPE: {param_metrics['mape']:.2f}%")
        
        report_lines.append("\nSpectrum Reconstruction:")
        spec_metrics = gen_results['spectrum_reconstruction']
        report_lines.append(f"  - MSE: {spec_metrics['mse']:.6f}")
        report_lines.append(f"  - MAE: {spec_metrics['mae']:.6f}")
        report_lines.append(f"  - R²: {spec_metrics['r2']:.4f}")
        report_lines.append(f"  - Pearson R: {spec_metrics['pearson_r']:.4f}")
        
        report_lines.append("\nMetrics Prediction:")
        metrics_metrics = gen_results['metrics_prediction']
        report_lines.append(f"  - MSE: {metrics_metrics['mse']:.6f}")
        report_lines.append(f"  - MAE: {metrics_metrics['mae']:.6f}")
        report_lines.append(f"  - R²: {metrics_metrics['r2']:.4f}")
        report_lines.append(f"  - Pearson R: {metrics_metrics['pearson_r']:.4f}")
        report_lines.append("")
        
        # 判别器评估
        disc_results = self.evaluation_results['discriminator_evaluation']
        report_lines.append("2. DISCRIMINATOR EVALUATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Overall Accuracy: {disc_results['overall_accuracy']:.4f}")
        report_lines.append(f"Real Sample Accuracy: {disc_results['real_accuracy']:.4f}")
        report_lines.append(f"Fake Sample Accuracy: {disc_results['fake_accuracy']:.4f}")
        report_lines.append(f"Total Loss: {disc_results['total_loss']:.6f}")
        report_lines.append(f"Real Score Mean±Std: {disc_results['real_score_mean']:.4f}±{disc_results['real_score_std']:.4f}")
        report_lines.append(f"Fake Score Mean±Std: {disc_results['fake_score_mean']:.4f}±{disc_results['fake_score_std']:.4f}")
        report_lines.append("")
        
        # 物理一致性评估
        phys_results = self.evaluation_results['physics_consistency']
        report_lines.append("3. PHYSICS CONSISTENCY EVALUATION")
        report_lines.append("-" * 40)
        report_lines.append(f"Parameter Range Violation Rate: {phys_results['param_range_violation_rate']:.4f}")
        report_lines.append(f"Average Violations per Sample: {phys_results['avg_param_violations_per_sample']:.4f}")
        report_lines.append(f"Spectrum Smoothness: {phys_results['spectrum_smoothness_mean']:.6f}±{phys_results['spectrum_smoothness_std']:.6f}")
        report_lines.append(f"Frequency Response: {phys_results['frequency_response_mean']:.6f}±{phys_results['frequency_response_std']:.6f}")
        report_lines.append("")
        
        # 总结
        report_lines.append("4. SUMMARY")
        report_lines.append("-" * 40)
        if param_metrics['r2'] > 0.8:
            report_lines.append("✓ Generator shows GOOD parameter prediction performance")
        elif param_metrics['r2'] > 0.6:
            report_lines.append("⚠ Generator shows MODERATE parameter prediction performance")
        else:
            report_lines.append("✗ Generator shows POOR parameter prediction performance")
        
        if disc_results['overall_accuracy'] > 0.8:
            report_lines.append("✓ Discriminator shows GOOD discrimination performance")
        elif disc_results['overall_accuracy'] > 0.6:
            report_lines.append("⚠ Discriminator shows MODERATE discrimination performance")
        else:
            report_lines.append("✗ Discriminator shows POOR discrimination performance")
        
        if phys_results['param_range_violation_rate'] < 0.1:
            report_lines.append("✓ Physics constraints are well satisfied")
        elif phys_results['param_range_violation_rate'] < 0.3:
            report_lines.append("⚠ Physics constraints are moderately satisfied")
        else:
            report_lines.append("✗ Physics constraints are poorly satisfied")
        
        report_lines.append("=" * 80)
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        if save_path is None:
            save_path = os.path.join(cfg.SAVED_MODELS_DIR, "evaluation_report.txt")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Evaluation report saved to: {save_path}")
        return report_content
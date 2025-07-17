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
        }\n        \n        print(f\"Generator evaluation completed with {len(all_real_params)} samples\")\n        return results\n    \n    def evaluate_discriminator(self, num_samples: int = 1000) -> Dict[str, Any]:\n        \"\"\"\n        评估判别器性能\n        \n        Args:\n            num_samples: 评估样本数\n            \n        Returns:\n            Dict[str, Any]: 评估结果\n        \"\"\"\n        print(f\"Evaluating Discriminator with {num_samples} samples...\")\n        \n        if self.discriminator is None or self.generator is None or self.dataset is None:\n            raise ValueError(\"Models and dataset must be loaded first!\")\n        \n        # 随机采样\n        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)\n        subset = Subset(self.dataset, sample_indices)\n        dataloader = DataLoader(subset, batch_size=64, shuffle=False)\n        \n        all_real_scores = []\n        all_fake_scores = []\n        bce_criterion = criterion_bce()\n        \n        with torch.no_grad():\n            for batch in dataloader:\n                real_spectrum, real_params_denorm, real_params_norm, _, _ = batch\n                \n                real_spectrum = real_spectrum.to(self.device)\n                real_params_denorm = real_params_denorm.to(self.device)\n                real_params_norm = real_params_norm.to(self.device)\n                \n                batch_size = real_spectrum.size(0)\n                \n                # 真实样本评分\n                real_scores = self.discriminator(real_spectrum, real_params_denorm)\n                all_real_scores.append(real_scores.cpu().numpy())\n                \n                # 生成假样本\n                fake_params_norm = self.generator(real_spectrum)\n                fake_params_denorm = denormalize_params(fake_params_norm, self.dataset.param_ranges)\n                \n                # 假样本评分\n                fake_scores = self.discriminator(real_spectrum, fake_params_denorm)\n                all_fake_scores.append(fake_scores.cpu().numpy())\n        \n        # 合并结果\n        all_real_scores = np.concatenate(all_real_scores, axis=0)\n        all_fake_scores = np.concatenate(all_fake_scores, axis=0)\n        \n        # 计算判别器性能指标\n        real_labels = np.ones_like(all_real_scores)\n        fake_labels = np.zeros_like(all_fake_scores)\n        \n        # 准确率计算\n        real_accuracy = np.mean(all_real_scores > 0.5)\n        fake_accuracy = np.mean(all_fake_scores < 0.5)\n        overall_accuracy = (real_accuracy + fake_accuracy) / 2\n        \n        # 损失计算\n        real_loss = -np.mean(np.log(all_real_scores + 1e-8))\n        fake_loss = -np.mean(np.log(1 - all_fake_scores + 1e-8))\n        total_loss = real_loss + fake_loss\n        \n        results = {\n            'real_accuracy': real_accuracy,\n            'fake_accuracy': fake_accuracy,\n            'overall_accuracy': overall_accuracy,\n            'real_loss': real_loss,\n            'fake_loss': fake_loss,\n            'total_loss': total_loss,\n            'real_score_mean': np.mean(all_real_scores),\n            'fake_score_mean': np.mean(all_fake_scores),\n            'real_score_std': np.std(all_real_scores),\n            'fake_score_std': np.std(all_fake_scores),\n            'num_samples': len(all_real_scores)\n        }\n        \n        print(f\"Discriminator evaluation completed with {len(all_real_scores)} samples\")\n        return results\n    \n    def evaluate_physics_consistency(self, num_samples: int = 1000) -> Dict[str, Any]:\n        \"\"\"\n        评估物理一致性\n        \n        Args:\n            num_samples: 评估样本数\n            \n        Returns:\n            Dict[str, Any]: 物理一致性评估结果\n        \"\"\"\n        print(f\"Evaluating Physics Consistency with {num_samples} samples...\")\n        \n        if self.generator is None or self.forward_model is None or self.dataset is None:\n            raise ValueError(\"Models and dataset must be loaded first!\")\n        \n        # 随机采样\n        sample_indices = np.random.choice(len(self.dataset), min(num_samples, len(self.dataset)), replace=False)\n        subset = Subset(self.dataset, sample_indices)\n        dataloader = DataLoader(subset, batch_size=64, shuffle=False)\n        \n        param_range_violations = []\n        spectrum_smoothness_scores = []\n        frequency_responses = []\n        \n        with torch.no_grad():\n            for batch in dataloader:\n                real_spectrum, _, _, _, _ = batch\n                real_spectrum = real_spectrum.to(self.device)\n                \n                # 生成器预测参数\n                pred_params_norm = self.generator(real_spectrum)\n                \n                # 检查参数范围约束\n                range_violations = torch.sum((pred_params_norm < 0) | (pred_params_norm > 1), dim=1).cpu().numpy()\n                param_range_violations.extend(range_violations)\n                \n                # 前向模型预测\n                pred_spectrum, pred_metrics = self.forward_model(pred_params_norm)\n                \n                # 光谱平滑性评估\n                spectrum_diff = torch.diff(pred_spectrum, dim=1)\n                smoothness = -torch.mean(spectrum_diff ** 2, dim=1).cpu().numpy()\n                spectrum_smoothness_scores.extend(smoothness)\n                \n                # 频率响应分析\n                freq_response = torch.mean(pred_spectrum, dim=1).cpu().numpy()\n                frequency_responses.extend(freq_response)\n        \n        # 统计结果\n        param_range_violations = np.array(param_range_violations)\n        spectrum_smoothness_scores = np.array(spectrum_smoothness_scores)\n        frequency_responses = np.array(frequency_responses)\n        \n        results = {\n            'param_range_violation_rate': np.mean(param_range_violations > 0),\n            'avg_param_violations_per_sample': np.mean(param_range_violations),\n            'spectrum_smoothness_mean': np.mean(spectrum_smoothness_scores),\n            'spectrum_smoothness_std': np.std(spectrum_smoothness_scores),\n            'frequency_response_mean': np.mean(frequency_responses),\n            'frequency_response_std': np.std(frequency_responses),\n            'num_samples': len(param_range_violations)\n        }\n        \n        print(f\"Physics consistency evaluation completed with {len(param_range_violations)} samples\")\n        return results\n    \n    def comprehensive_evaluation(self, num_samples: int = 1000) -> Dict[str, Any]:\n        \"\"\"\n        执行全面评估\n        \n        Args:\n            num_samples: 评估样本数\n            \n        Returns:\n            Dict[str, Any]: 完整评估结果\n        \"\"\"\n        print(\"\\n=== Starting Comprehensive Evaluation ===\")\n        start_time = time.time()\n        \n        # 检查模型和数据集\n        if not all([self.generator, self.discriminator, self.forward_model, self.dataset]):\n            raise ValueError(\"Models and dataset must be loaded first!\")\n        \n        # 执行各项评估\n        results = {\n            'generator_evaluation': self.evaluate_generator(num_samples),\n            'discriminator_evaluation': self.evaluate_discriminator(num_samples),\n            'physics_consistency': self.evaluate_physics_consistency(num_samples),\n            'evaluation_time': time.time() - start_time,\n            'num_samples': num_samples\n        }\n        \n        # 保存结果\n        self.evaluation_results = results\n        \n        print(f\"\\n=== Comprehensive Evaluation Completed in {results['evaluation_time']:.2f}s ===\")\n        return results\n    \n    def save_evaluation_results(self, save_path: str = None) -> None:\n        \"\"\"\n        保存评估结果\n        \n        Args:\n            save_path: 保存路径\n        \"\"\"\n        if save_path is None:\n            save_path = os.path.join(cfg.SAVED_MODELS_DIR, \"evaluation_results.npz\")\n        \n        # 准备保存数据（移除不能序列化的数据）\n        save_data = {}\n        for key, value in self.evaluation_results.items():\n            if key != 'generator_evaluation' or 'data' not in value:\n                save_data[key] = value\n            else:\n                # 保存生成器评估结果但排除原始数据\n                save_data[key] = {k: v for k, v in value.items() if k != 'data'}\n        \n        np.savez(save_path, **save_data)\n        print(f\"Evaluation results saved to: {save_path}\")\n    \n    def generate_report(self, save_path: str = None) -> str:\n        \"\"\"\n        生成评估报告\n        \n        Args:\n            save_path: 报告保存路径\n            \n        Returns:\n            str: 报告内容\n        \"\"\"\n        if not self.evaluation_results:\n            raise ValueError(\"No evaluation results available. Run comprehensive_evaluation first.\")\n        \n        report_lines = []\n        report_lines.append(\"=\" * 80)\n        report_lines.append(\"PI-GAN MODEL EVALUATION REPORT\")\n        report_lines.append(\"=\" * 80)\n        \n        # 基本信息\n        report_lines.append(f\"Evaluation Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\")\n        report_lines.append(f\"Number of Samples: {self.evaluation_results['num_samples']}\")\n        report_lines.append(f\"Evaluation Time: {self.evaluation_results['evaluation_time']:.2f}s\")\n        report_lines.append(\"\")\n        \n        # 生成器评估\n        gen_results = self.evaluation_results['generator_evaluation']\n        report_lines.append(\"1. GENERATOR EVALUATION\")\n        report_lines.append(\"-\" * 40)\n        \n        report_lines.append(\"Parameter Prediction:\")\n        param_metrics = gen_results['parameter_prediction']\n        report_lines.append(f\"  - MSE: {param_metrics['mse']:.6f}\")\n        report_lines.append(f\"  - MAE: {param_metrics['mae']:.6f}\")\n        report_lines.append(f\"  - RMSE: {param_metrics['rmse']:.6f}\")\n        report_lines.append(f\"  - R²: {param_metrics['r2']:.4f}\")\n        report_lines.append(f\"  - Pearson R: {param_metrics['pearson_r']:.4f}\")\n        report_lines.append(f\"  - MAPE: {param_metrics['mape']:.2f}%\")\n        \n        report_lines.append(\"\\nSpectrum Reconstruction:\")\n        spec_metrics = gen_results['spectrum_reconstruction']\n        report_lines.append(f\"  - MSE: {spec_metrics['mse']:.6f}\")\n        report_lines.append(f\"  - MAE: {spec_metrics['mae']:.6f}\")\n        report_lines.append(f\"  - R²: {spec_metrics['r2']:.4f}\")\n        report_lines.append(f\"  - Pearson R: {spec_metrics['pearson_r']:.4f}\")\n        \n        report_lines.append(\"\\nMetrics Prediction:\")\n        metrics_metrics = gen_results['metrics_prediction']\n        report_lines.append(f\"  - MSE: {metrics_metrics['mse']:.6f}\")\n        report_lines.append(f\"  - MAE: {metrics_metrics['mae']:.6f}\")\n        report_lines.append(f\"  - R²: {metrics_metrics['r2']:.4f}\")\n        report_lines.append(f\"  - Pearson R: {metrics_metrics['pearson_r']:.4f}\")\n        report_lines.append(\"\")\n        \n        # 判别器评估\n        disc_results = self.evaluation_results['discriminator_evaluation']\n        report_lines.append(\"2. DISCRIMINATOR EVALUATION\")\n        report_lines.append(\"-\" * 40)\n        report_lines.append(f\"Overall Accuracy: {disc_results['overall_accuracy']:.4f}\")\n        report_lines.append(f\"Real Sample Accuracy: {disc_results['real_accuracy']:.4f}\")\n        report_lines.append(f\"Fake Sample Accuracy: {disc_results['fake_accuracy']:.4f}\")\n        report_lines.append(f\"Total Loss: {disc_results['total_loss']:.6f}\")\n        report_lines.append(f\"Real Score Mean±Std: {disc_results['real_score_mean']:.4f}±{disc_results['real_score_std']:.4f}\")\n        report_lines.append(f\"Fake Score Mean±Std: {disc_results['fake_score_mean']:.4f}±{disc_results['fake_score_std']:.4f}\")\n        report_lines.append(\"\")\n        \n        # 物理一致性评估\n        phys_results = self.evaluation_results['physics_consistency']\n        report_lines.append(\"3. PHYSICS CONSISTENCY EVALUATION\")\n        report_lines.append(\"-\" * 40)\n        report_lines.append(f\"Parameter Range Violation Rate: {phys_results['param_range_violation_rate']:.4f}\")\n        report_lines.append(f\"Average Violations per Sample: {phys_results['avg_param_violations_per_sample']:.4f}\")\n        report_lines.append(f\"Spectrum Smoothness: {phys_results['spectrum_smoothness_mean']:.6f}±{phys_results['spectrum_smoothness_std']:.6f}\")\n        report_lines.append(f\"Frequency Response: {phys_results['frequency_response_mean']:.6f}±{phys_results['frequency_response_std']:.6f}\")\n        report_lines.append(\"\")\n        \n        # 总结\n        report_lines.append(\"4. SUMMARY\")\n        report_lines.append(\"-\" * 40)\n        if param_metrics['r2'] > 0.8:\n            report_lines.append(\"✓ Generator shows GOOD parameter prediction performance\")\n        elif param_metrics['r2'] > 0.6:\n            report_lines.append(\"⚠ Generator shows MODERATE parameter prediction performance\")\n        else:\n            report_lines.append(\"✗ Generator shows POOR parameter prediction performance\")\n        \n        if disc_results['overall_accuracy'] > 0.8:\n            report_lines.append(\"✓ Discriminator shows GOOD discrimination performance\")\n        elif disc_results['overall_accuracy'] > 0.6:\n            report_lines.append(\"⚠ Discriminator shows MODERATE discrimination performance\")\n        else:\n            report_lines.append(\"✗ Discriminator shows POOR discrimination performance\")\n        \n        if phys_results['param_range_violation_rate'] < 0.1:\n            report_lines.append(\"✓ Physics constraints are well satisfied\")\n        elif phys_results['param_range_violation_rate'] < 0.3:\n            report_lines.append(\"⚠ Physics constraints are moderately satisfied\")\n        else:\n            report_lines.append(\"✗ Physics constraints are poorly satisfied\")\n        \n        report_lines.append(\"=\" * 80)\n        \n        report_content = \"\\n\".join(report_lines)\n        \n        # 保存报告\n        if save_path is None:\n            save_path = os.path.join(cfg.SAVED_MODELS_DIR, \"evaluation_report.txt\")\n        \n        with open(save_path, 'w', encoding='utf-8') as f:\n            f.write(report_content)\n        \n        print(f\"Evaluation report saved to: {save_path}\")\n        return report_content
import sys
import os
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import DataLoader, Subset
import time
import argparse

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import models
from core.models.generator import EnhancedGenerator
from core.models.discriminator import EnhancedDiscriminator
from core.models.forward_model import EnhancedForwardPINN

import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics, normalize_spectrum
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_mse, criterion_bce
from core.utils.visualization import EvaluationVisualizer

# Import the new evaluation modules
from core.evaluate.evaluate_forward_model import evaluate_forward_model
from core.evaluate.evaluate_pigan import evaluate_pigan, evaluate_pigan_enhanced

class ComprehensiveEvaluator:
    """
    综合评估器：协调前向网络评估、PI-GAN评估、结构预测和模型验证
    """
    
    def __init__(self, device: str = "auto"):
        """
        初始化综合评估器
        
        Args:
            device: 计算设备 ("auto", "cpu", "cuda")
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.generator = None
        self.discriminator = None
        self.forward_model = None
        self.dataset = None # Add dataset attribute
        self.evaluation_results = {}
        
        # Initialize visualizer
        plots_dir = os.path.join(cfg.PROJECT_ROOT, "plots")
        self.visualizer = EvaluationVisualizer(save_dir=plots_dir)
        
        print(f"Comprehensive Evaluator initialized on device: {self.device}")
    
    def load_models(self, model_dir: str = None) -> bool:
        """
        加载训练好的增强模型
        
        Args:
            model_dir: 模型保存目录
            
        Returns:
            bool: 加载是否成功
        """
        if model_dir is None:
            model_dir = cfg.SAVED_MODELS_DIR
            
        print(f"Loading enhanced models from: {model_dir}")
        
        try:
            # Initialize models
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
            
            # Load weights
            gen_path = os.path.join(model_dir, "generator_final.pth")
            disc_path = os.path.join(model_dir, "discriminator_final.pth")
            fwd_path = os.path.join(model_dir, "forward_model_final.pth")
                
            self.generator.load_state_dict(torch.load(gen_path, map_location=self.device))
            self.discriminator.load_state_dict(torch.load(disc_path, map_location=self.device))
            self.forward_model.load_state_dict(torch.load(fwd_path, map_location=self.device))
            
            # Set to evaluation mode
            self.generator.eval()
            self.discriminator.eval()
            self.forward_model.eval()
            
            print("✓ Models loaded successfully!")
            return True
            
        except Exception as e:
            print(f"✗ Error loading models: {e}")
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
        
        # Randomly sample
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
                
                # Generator predicts parameters
                if isinstance(self.generator, EnhancedGenerator):
                    latent_vector = torch.randn(real_spectrum.shape[0], cfg.Z_DIM).to(self.device)
                    pred_params_norm = self.generator(real_spectrum, latent_vector)
                else:
                    pred_params_norm = self.generator(real_spectrum)
                
                # Check parameter range constraints
                range_violations = torch.sum((pred_params_norm < 0) | (pred_params_norm > 1), dim=1).cpu().numpy()
                param_range_violations.extend(range_violations)
                
                # Forward model reconstructs spectrum
                if isinstance(self.forward_model, EnhancedForwardPINN):
                    recon_spectrum, _ = self.forward_model(pred_params_norm)
                else:
                    recon_spectrum, _ = self.forward_model(pred_params_norm)
                
                # Reconstruction error
                recon_error = torch.mean((real_spectrum - recon_spectrum) ** 2, dim=1).cpu().numpy()
                reconstruction_errors.extend(recon_error)
                
                # Consistency score (1 - normalized reconstruction error)
                consistency = 1.0 / (1.0 + recon_error)
                consistency_scores.extend(consistency)
        
        # Statistical results
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
        
        # Randomly sample
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
                
                # Cycle consistency test: spectrum -> params -> spectrum
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
                
                # Prediction stability test: consistency after adding small noise
                noise = torch.randn_like(real_spectrum) * 0.01
                noisy_spectrum = real_spectrum + noise
                if isinstance(self.generator, EnhancedGenerator):
                    pred_params_noisy = self.generator(noisy_spectrum, latent_vector)
                else:
                    pred_params_noisy = self.generator(noisy_spectrum)
                
                stability = torch.mean((pred_params_norm - pred_params_noisy) ** 2, dim=1).cpu().numpy()
                prediction_stability.extend(stability)
                
                # Physical plausibility: degree of satisfying physical constraints for predicted parameters
                pred_params_denorm = denormalize_params(pred_params_norm, self.dataset.param_ranges)
                
                # Check if parameters are within reasonable range
                plausibility_score = torch.mean(
                    torch.sigmoid(pred_params_norm * 10 - 5), dim=1
                ).cpu().numpy()
                physical_plausibility.extend(plausibility_score)
        
        # Statistical results
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
    
    def run_comprehensive_evaluation(self, num_samples: int = 1000, model_dir: str = None, data_path: str = None) -> Dict[str, Any]:
        """
        运行全面评估
        
        Args:
            num_samples: 评估样本数
            model_dir: 模型目录路径
            data_path: 数据集路径
            
        Returns:
            Dict[str, Any]: 完整评估结果
        """
        print("\n" + "="*80)
        print("PI-GAN COMPREHENSIVE EVALUATION")
        print("="*80)
        
        start_time = time.time()
        
        # Load models and dataset if not already loaded
        if not self.generator or not self.discriminator or not self.forward_model:
            if not self.load_models(model_dir):
                raise ValueError("Failed to load models for comprehensive evaluation!")
        if not self.dataset:
            if not self.load_dataset(data_path):
                raise ValueError("Failed to load dataset for comprehensive evaluation!")

        # Execute evaluations from separate modules
        fwd_eval_results = evaluate_forward_model(config_path='config/config.py', num_samples=num_samples)
        pigan_eval_results = evaluate_pigan(model_dir=model_dir, data_path=data_path, num_samples=num_samples)

        results = {
            'forward_network_evaluation': fwd_eval_results,
            'pigan_evaluation': pigan_eval_results,
            'structural_prediction_evaluation': self.evaluate_structural_prediction(min(num_samples//2, 500)),
            'model_validation': self.evaluate_model_validation(min(num_samples//2, 500)),
            'evaluation_time': time.time() - start_time,
            'total_samples': num_samples
        }
        
        # Save results
        self.evaluation_results = results
        
        # Generate visualizations
        print(f"\n🎨 Generating evaluation visualizations...")
        self.generate_visualizations(results)
        
        print(f"\n" + "="*80)
        print(f"EVALUATION COMPLETED in {results['evaluation_time']:.2f}s")
        print("="*80)
        
        return results
    
    def run_comprehensive_evaluation_enhanced(self, num_samples: int = 1000, model_dir: str = None, data_path: str = None) -> Dict[str, Any]:
        """
        运行增强版全面评估
        
        Args:
            num_samples: 评估样本数
            model_dir: 模型目录路径
            data_path: 数据集路径
            
        Returns:
            Dict[str, Any]: 完整评估结果
        """
        print("\n" + "="*80)
        print("ENHANCED PI-GAN COMPREHENSIVE EVALUATION")
        print("="*80)
        
        start_time = time.time()
        
        # Load models and dataset if not already loaded
        if not self.generator or not self.discriminator or not self.forward_model:
            if not self.load_models(model_dir):
                raise ValueError("Failed to load models for comprehensive evaluation!")
        if not self.dataset:
            if not self.load_dataset(data_path):
                raise ValueError("Failed to load dataset for comprehensive evaluation!")

        # Execute evaluations from separate modules
        fwd_eval_results = evaluate_forward_model(config_path='config/config.py', num_samples=num_samples)
        pigan_eval_results = evaluate_pigan_enhanced(model_dir=model_dir, data_path=data_path, num_samples=num_samples)
        
        results = {
            'forward_network_evaluation': fwd_eval_results,
            'pigan_evaluation': pigan_eval_results,
            'structural_prediction_evaluation': self.evaluate_structural_prediction(min(num_samples//2, 500)),
            'model_validation': self.evaluate_model_validation(min(num_samples//2, 500)),
            'evaluation_time': time.time() - start_time,
            'total_samples': num_samples
        }
        
        # Save results
        self.evaluation_results = results
        
        # Generate visualizations
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
        
        # Check for physics consistency evaluation
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
        
        # 3. Structural prediction evaluation
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
        
        # 4. Model validation
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
        
        # Summary
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
        
        # Save report
        if save_path is None:
            save_path = os.path.join(self.visualizer.save_dir, "unified_evaluation_report.txt")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\nEvaluation report saved to: {save_path}")
        return report_content

# Main function
def main():
    parser = argparse.ArgumentParser(description="Run comprehensive PI-GAN evaluation")
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
    parser.add_argument('--enhanced', action='store_true',
                        help='Whether to run enhanced evaluation')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create evaluator
    evaluator = ComprehensiveEvaluator(device=args.device)
    
    # Load models and data (now handled within run_comprehensive_evaluation methods)
    
    # Run evaluation
    if args.enhanced:
        results = evaluator.run_comprehensive_evaluation_enhanced(
            num_samples=args.num_samples,
            model_dir=args.model_dir,
            data_path=args.data_path
        )
    else:
        results = evaluator.run_comprehensive_evaluation(
            num_samples=args.num_samples,
            model_dir=args.model_dir,
            data_path=args.data_path
        )
    
    # Generate report
    evaluator.generate_summary_report()
    
    print("\n✅ Comprehensive evaluation completed successfully!")

if __name__ == "__main__":
    main()
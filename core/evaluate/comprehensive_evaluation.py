# PI_GAN_THZ/core/evaluate/comprehensive_evaluation.py

import sys
import os
import argparse
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from enhanced_evaluator import EnhancedEvaluator
from enhanced_predict import EnhancedPredictor
import config.config as cfg
from core.utils.set_seed import set_seed

def run_comprehensive_evaluation(model_dir: str = None, 
                                data_path: str = None,
                                num_samples: int = 1000,
                                save_results: bool = True,
                                generate_report: bool = True) -> None:
    """
    运行全面的模型评估
    
    Args:
        model_dir: 模型目录
        data_path: 数据路径
        num_samples: 评估样本数
        save_results: 是否保存结果
        generate_report: 是否生成报告
    """
    print("\\n" + "="*80)
    print("PI-GAN COMPREHENSIVE EVALUATION")
    print("="*80)
    
    # 设置随机种子
    set_seed(cfg.RANDOM_SEED)
    
    # 创建评估器
    evaluator = EnhancedEvaluator()
    
    # 加载模型和数据
    print("\\n1. Loading models and dataset...")
    if not evaluator.load_models(model_dir):
        print("❌ Failed to load models!")
        return
    
    if not evaluator.load_dataset(data_path):
        print("❌ Failed to load dataset!")
        return
    
    print("✅ Models and dataset loaded successfully!")
    
    # 执行全面评估
    print(f"\\n2. Running comprehensive evaluation with {num_samples} samples...")
    try:
        results = evaluator.comprehensive_evaluation(num_samples)
        print("✅ Comprehensive evaluation completed!")
        
        # 打印关键结果
        print_evaluation_summary(results)
        
        # 保存结果
        if save_results:
            print("\\n3. Saving evaluation results...")
            evaluator.save_evaluation_results()
            print("✅ Results saved!")
        
        # 生成报告
        if generate_report:
            print("\\n4. Generating evaluation report...")
            report = evaluator.generate_report()
            print("✅ Report generated!")
            
            # 打印报告摘要
            print("\\n" + "="*50)
            print("EVALUATION REPORT SUMMARY")
            print("="*50)
            print(report[-1000:])  # 打印报告末尾的总结部分
        
        print("\\n🎉 Comprehensive evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

def print_evaluation_summary(results: dict) -> None:
    """
    打印评估结果摘要
    
    Args:
        results: 评估结果字典
    """
    print("\\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    # 生成器评估
    gen_results = results['generator_evaluation']
    print("\\n📊 Generator Performance:")
    print(f"  Parameter Prediction R²: {gen_results['parameter_prediction']['r2']:.4f}")
    print(f"  Parameter Prediction RMSE: {gen_results['parameter_prediction']['rmse']:.6f}")
    print(f"  Spectrum Reconstruction R²: {gen_results['spectrum_reconstruction']['r2']:.4f}")
    print(f"  Metrics Prediction R²: {gen_results['metrics_prediction']['r2']:.4f}")
    
    # 判别器评估
    disc_results = results['discriminator_evaluation']
    print("\\n🎯 Discriminator Performance:")
    print(f"  Overall Accuracy: {disc_results['overall_accuracy']:.4f}")
    print(f"  Real Sample Accuracy: {disc_results['real_accuracy']:.4f}")
    print(f"  Fake Sample Accuracy: {disc_results['fake_accuracy']:.4f}")
    
    # 物理一致性
    phys_results = results['physics_consistency']
    print("\\n⚗️ Physics Consistency:")
    print(f"  Parameter Range Violation Rate: {phys_results['param_range_violation_rate']:.4f}")
    print(f"  Average Violations per Sample: {phys_results['avg_param_violations_per_sample']:.4f}")
    
    print(f"\\n⏱️ Total Evaluation Time: {results['evaluation_time']:.2f}s")
    print(f"📈 Number of Samples Evaluated: {results['num_samples']}")

def run_prediction_test(model_dir: str = None,
                       test_spectrum_path: str = None,
                       uncertainty_samples: int = 100) -> None:
    """
    运行预测测试
    
    Args:
        model_dir: 模型目录
        test_spectrum_path: 测试光谱路径
        uncertainty_samples: 不确定性采样次数
    """
    print("\\n" + "="*50)
    print("PREDICTION TEST")
    print("="*50)
    
    # 创建预测器
    predictor = EnhancedPredictor()
    
    # 加载模型
    if not predictor.load_models(model_dir):
        print("❌ Failed to load models for prediction!")
        return
    
    if not predictor.load_dataset():
        print("❌ Failed to load dataset for prediction!")
        return
    
    # 测试预测
    if test_spectrum_path and os.path.exists(test_spectrum_path):
        print(f"\\n🔬 Testing prediction with: {test_spectrum_path}")
        try:
            results = predictor.predict_structure(
                spectrum_input=test_spectrum_path,
                uncertainty_samples=uncertainty_samples,
                optimize_prediction=True
            )
            
            print("\\n✅ Prediction completed!")
            print(f"Predicted Parameters: {results['predicted_params_denorm'][0]}")
            print(f"Reconstruction MSE: {results['reconstruction_mse']:.6f}")
            print(f"Prediction Time: {results['prediction_time']:.2f}s")
            
            if results.get('uncertainty_available'):
                print(f"Uncertainty estimation completed with {results['num_uncertainty_samples']} samples")
            
            # 可视化结果
            predictor.visualize_prediction(results)
            predictor.save_prediction_results(results)
            
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
    else:
        print(f"⚠️ Test spectrum file not found: {test_spectrum_path}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="PI-GAN Comprehensive Evaluation")
    parser.add_argument('--model_dir', type=str, default=None,
                        help='Model directory path')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Dataset path')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples for evaluation')
    parser.add_argument('--test_spectrum', type=str, default=None,
                        help='Path to test spectrum file for prediction test')
    parser.add_argument('--uncertainty_samples', type=int, default=100,
                        help='Number of uncertainty samples for prediction test')
    parser.add_argument('--skip_evaluation', action='store_true',
                        help='Skip comprehensive evaluation')
    parser.add_argument('--skip_prediction', action='store_true',
                        help='Skip prediction test')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--no_report', action='store_true',
                        help='Do not generate report')
    
    args = parser.parse_args()
    
    # 创建输出目录
    cfg.create_directories()
    
    start_time = time.time()
    
    # 运行全面评估
    if not args.skip_evaluation:
        run_comprehensive_evaluation(
            model_dir=args.model_dir,
            data_path=args.data_path,
            num_samples=args.num_samples,
            save_results=not args.no_save,
            generate_report=not args.no_report
        )
    
    # 运行预测测试
    if not args.skip_prediction:
        # 如果没有提供测试光谱，尝试使用默认路径
        test_spectrum = args.test_spectrum
        if test_spectrum is None:
            # 尝试一些默认路径
            possible_paths = [
                os.path.join(cfg.DATA_DIR, "THZ.txt"),
                os.path.join(cfg.PROJECT_ROOT, "dataset", "THZ.txt"),
                os.path.join(cfg.PROJECT_ROOT, "test_spectrum.txt")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    test_spectrum = path
                    break
        
        if test_spectrum:
            run_prediction_test(
                model_dir=args.model_dir,
                test_spectrum_path=test_spectrum,
                uncertainty_samples=args.uncertainty_samples
            )
        else:
            print("\\n⚠️ No test spectrum provided, skipping prediction test")
            print("   Use --test_spectrum to specify a test file")
    
    total_time = time.time() - start_time
    print(f"\\n🏁 Total evaluation time: {total_time:.2f}s")
    print("\\n✨ All evaluations completed!")

if __name__ == "__main__":
    main()
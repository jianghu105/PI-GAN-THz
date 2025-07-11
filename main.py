# PI_GAN_THZ/main.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import argparse # 导入 argparse 库

import config.config as cfg
from core.utils.logger import Logger

# 导入所有模型
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics, normalize_spectrum # 确保导入 normalize_spectrum
from core.models.forward_model import ForwardModel
from core.models.generator import Generator
from core.models.discriminator import Discriminator

# 导入训练函数
from core.train.pretrain_fwd_model import pretrain_forward_model
from core.train.train_pigan import train_pigan

# 导入评估和可视化函数
from core.utils.eval import load_model_state, evaluate_pigan_inverse_design
from core.utils.visual import plot_spectrum_comparison, plot_parameter_prediction, \
                               plot_metrics_prediction, plot_spectrum_uncertainty

def set_seed(seed):
    """设置所有随机源的种子以确保复现性。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    # --- 1. 定义命令行参数 ---
    parser = argparse.ArgumentParser(description="PI-GAN for THz Metamaterial Inverse Design")

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'eval', 'predict'],
                        help="Operation mode: 'train' for training, 'eval' for evaluation, 'predict' for inference on new data.")
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=None,
                        help="Number of training epochs. Overrides config if specified.")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size for training. Overrides config if specified.")
    parser.add_argument('--lr_g', type=float, default=None,
                        help="Learning rate for Generator. Overrides config if specified.")
    parser.add_argument('--lr_d', type=float, default=None,
                        help="Learning rate for Discriminator. Overrides config if specified.")
    parser.add_argument('--lr_fwd_sim', type=float, default=None,
                        help="Learning rate for Forward Model pre-training. Overrides config if specified.")
    parser.add_argument('--log_interval', type=int, default=None,
                        help="Logging interval (epochs). Overrides config if specified.")
    parser.add_argument('--save_interval', type=int, default=None,
                        help="Checkpoint saving interval (epochs). Overrides config if specified.")
    
    # 加载模型/检查点
    parser.add_argument('--load_generator', type=str, default=None,
                        help="Path to saved Generator model (.pth) to load for eval/predict or continued training.")
    parser.add_argument('--load_discriminator', type=str, default=None,
                        help="Path to saved Discriminator model (.pth) to load for continued training.")
    parser.add_argument('--load_forward_model', type=str, default=None,
                        help="Path to saved Forward Model (.pth) to load for eval/predict or continued training.")
    parser.add_argument('--load_checkpoint', type=str, default=None,
                        help="Path to a full checkpoint (.pth) to resume training.")

    # 评估/预测相关参数
    parser.add_argument('--mc_dropout_samples', type=int, default=0,
                        help="Number of Monte Carlo Dropout samples for uncertainty estimation during eval/predict.")
    parser.add_argument('--plot_results', action='store_true',
                        help="Whether to save evaluation/prediction plots.")
    parser.add_argument('--num_eval_samples_to_plot', type=int, default=5,
                        help="Number of individual samples to plot during evaluation.")
    
    # 预测模式特有参数
    parser.add_argument('--input_spectrum_path', type=str, default=None,
                        help="Path to a CSV file containing new spectrum data for prediction mode.")
    parser.add_argument('--output_prediction_path', type=str, default=None,
                        help="Path to save prediction results (CSV format).")

    args = parser.parse_args()

    # --- 2. 覆盖配置文件中的参数 (如果命令行提供了) ---
    if args.epochs is not None: cfg.NUM_EPOCHS = args.epochs
    if args.batch_size is not None: cfg.BATCH_SIZE = args.batch_size
    if args.lr_g is not None: cfg.LR_G = args.lr_g
    if args.lr_d is not None: cfg.LR_D = args.lr_d
    if args.lr_fwd_sim is not None: cfg.LR_FWD_SIM = args.lr_fwd_sim
    if args.log_interval is not None: cfg.LOG_INTERVAL = args.log_interval
    if args.save_interval is not None: cfg.SAVE_INTERVAL = args.save_interval

    # --- 3. 初始化通用设置 ---
    cfg.create_directories() # 确保所有必要的目录存在
    logger = Logger(log_dir=cfg.LOG_DIR, experiment_name=f"PIGAN_{args.mode}")
    logger.info(f"Starting PI_GAN_THZ Project in '{args.mode}' mode.")
    logger.info(f"Using device: {cfg.DEVICE}")
    logger.info(f"Random seed set to: {cfg.RANDOM_SEED}")
    set_seed(cfg.RANDOM_SEED)

    device = torch.device(cfg.DEVICE)

    # 4. 加载数据 (只在训练和评估时需要完整数据集)
    if args.mode in ['train', 'eval']:
        dataset = MetamaterialDataset(
            data_path=cfg.FULL_DATA_PATH,
            num_points_per_sample=cfg.NUM_SPECTRUM_POINTS
        )
        num_workers = os.cpu_count() // 2 if os.cpu_count() else 0
        dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=True if args.mode == 'train' else False, num_workers=num_workers)
        logger.info(f"Dataset size: {len(dataset)} samples")
        logger.info(f"Number of batches: {len(dataloader)}")
    else: # Predict mode doesn't need full dataset for training/eval
        dataset = MetamaterialDataset(data_path=cfg.FULL_DATA_PATH, num_points_per_sample=cfg.NUM_SPECTRUM_POINTS,
                                      load_data=False) # Only load metadata like param_names, metric_names
        dataloader = None # Not needed for prediction unless input_spectrum_path points to test data

    # 5. 实例化模型
    generator = Generator(
        input_spectrum_dim=cfg.GENERATOR_INPUT_DIM,
        output_param_dim=cfg.GENERATOR_OUTPUT_DIM
    ).to(device)

    discriminator = Discriminator(
        input_spectrum_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM,
        input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM
    ).to(device)

    forward_model = ForwardModel(
        input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
        output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
        output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM
    ).to(device)

    logger.info("\n--- Model Architectures ---")
    logger.info(f"Generator:\n{generator}")
    logger.info(f"Discriminator:\n{discriminator}")
    logger.info(f"ForwardModel:\n{forward_model}")

    # --- 6. 根据模式执行操作 ---

    # 训练模式
    if args.mode == 'train':
        # 如果指定了检查点，则加载
        if args.load_checkpoint:
            try:
                checkpoint = torch.load(args.load_checkpoint, map_location=device)
                generator.load_state_dict(checkpoint['generator_state_dict'])
                discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                forward_model.load_state_dict(checkpoint['forward_model_state_dict'])
                # 还可以加载优化器状态等
                logger.info(f"Resuming training from checkpoint: {args.load_checkpoint}")
            except Exception as e:
                logger.error(f"Error loading checkpoint {args.load_checkpoint}: {e}")
                exit(1)
        else: # 如果没有从检查点恢复，则预训练前向模型
            # 预训练前向模型 (除非是从完整检查点恢复，检查点已包含预训练后的前向模型)
            pretrain_forward_model(
                forward_model=forward_model,
                dataloader=dataloader,
                device=device,
                num_epochs=cfg.NUM_EPOCHS, # 可以为前向模型预训练设置单独的 epoch 数
                lr=cfg.LR_FWD_SIM,
                logger=logger
            )

        train_pigan(
            dataloader=dataloader,
            device=device,
            generator=generator,
            discriminator=discriminator,
            forward_model=forward_model,
            dataset=dataset, # 传递 dataset 实例
            logger=logger
        )

    # 评估模式
    elif args.mode == 'eval':
        # 确保加载了模型
        if not args.load_generator or not args.load_forward_model:
            logger.error("For evaluation mode, --load_generator and --load_forward_model must be specified.")
            exit(1)
        
        load_model_state(generator, args.load_generator, device)
        load_model_state(forward_model, args.load_forward_model, device)

        logger.info("\n--- Starting Model Evaluation ---")
        # 创建一个新的 DataLoader 用于评估 (通常是测试集，这里为了演示，暂时用训练集)
        test_dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=num_workers)

        eval_results, real_spec, pred_spec, \
        real_params, pred_params, real_metrics, pred_metrics, \
        mc_spec_samples, mc_metrics_samples = \
            evaluate_pigan_inverse_design(
                generator,
                forward_model,
                test_dataloader,
                device,
                dataset,
                mc_dropout_samples=args.mc_dropout_samples
            )

        if args.plot_results:
            logger.info("\n--- Saving Evaluation Plots ---")
            visualization_save_dir = os.path.join(cfg.PROJECT_ROOT, "evaluation_plots", f"{logger.experiment_log_dir.split(os.sep)[-1]}")
            os.makedirs(visualization_save_dir, exist_ok=True)

            # 绘制几个样本的光谱对比图
            num_samples_to_plot = min(args.num_eval_samples_to_plot, len(real_spec))
            for i in range(num_samples_to_plot):
                plot_spectrum_comparison(real_spec[i], pred_spec[i], dataset.frequencies, i, save_dir=visualization_save_dir)
                if mc_spec_samples is not None and args.mc_dropout_samples > 0:
                    plot_spectrum_uncertainty(real_spec[i], mc_spec_samples[:, i, :], dataset.frequencies, i, save_dir=visualization_save_dir)

            plot_parameter_prediction(real_params, pred_params, dataset.param_names, save_dir=visualization_save_dir)
            plot_metrics_prediction(real_metrics, pred_metrics, dataset.metric_names, save_dir=visualization_save_dir)
            logger.info(f"Evaluation plots saved to: {visualization_save_dir}")
        
        logger.info("--- Model Evaluation Complete ---")

    # 预测模式
    elif args.mode == 'predict':
        # 确保加载了生成器和前向模型
        if not args.load_generator or not args.load_forward_model:
            logger.error("For prediction mode, --load_generator and --load_forward_model must be specified.")
            exit(1)
        
        load_model_state(generator, args.load_generator, device)
        load_model_state(forward_model, args.load_forward_model, device)

        if not args.input_spectrum_path:
            logger.error("For prediction mode, --input_spectrum_path must be specified.")
            exit(1)
        
        logger.info(f"\n--- Starting Prediction on {args.input_spectrum_path} ---")
        
        try:
            # 读取新的光谱数据 (假设是CSV，每行一个光谱)
            new_spectrum_data = np.loadtxt(args.input_spectrum_path, delimiter=',')
            if new_spectrum_data.ndim == 1: # 如果只有一行数据
                new_spectrum_data = new_spectrum_data.reshape(1, -1)
            
            if new_spectrum_data.shape[1] != cfg.NUM_SPECTRUM_POINTS:
                logger.error(f"Input spectrum points ({new_spectrum_data.shape[1]}) do not match expected ({cfg.NUM_SPECTRUM_POINTS}).")
                exit(1)

            # 归一化输入光谱
            normalized_input_spectrum = normalize_spectrum(torch.tensor(new_spectrum_data, dtype=torch.float32)).to(device)

            # 逆向设计：预测结构参数
            generator.eval()
            with torch.no_grad():
                predicted_params_norm = generator(normalized_input_spectrum)
            predicted_params_denorm = denormalize_params(predicted_params_norm, dataset.param_ranges).cpu().numpy()

            # 前向模拟：根据预测参数生成重建光谱和物理指标
            forward_model.eval()
            if args.mc_dropout_samples > 0:
                # 开启ForwardModel的Dropout层 (仅在评估时用于MC Dropout)
                for m in forward_model.modules():
                    if isinstance(m, nn.Dropout):
                        m.train() # 临时设置为训练模式以启用dropout

                mc_spec_preds = []
                mc_metrics_preds = []
                for _ in range(args.mc_dropout_samples):
                    recon_spectrum_mc, predicted_metrics_norm_mc = forward_model(predicted_params_norm)
                    mc_spec_preds.append(recon_spectrum_mc.cpu().numpy())
                    mc_metrics_preds.append(predicted_metrics_norm_mc.cpu().numpy())
                
                # 关闭ForwardModel的Dropout层
                for m in forward_model.modules():
                    if isinstance(m, nn.Dropout):
                        m.eval() # 恢复评估模式

                recon_spectrum_avg = np.mean(np.array(mc_spec_preds), axis=0)
                predicted_metrics_norm_avg = np.mean(np.array(mc_metrics_preds), axis=0)
                
                # 计算不确定性 (标准差)
                recon_spectrum_std = np.std(np.array(mc_spec_preds), axis=0)
                predicted_metrics_std = np.std(np.array(mc_metrics_preds), axis=0)
            else:
                with torch.no_grad():
                    recon_spectrum_avg, predicted_metrics_norm_avg = forward_model(predicted_params_norm)
                    recon_spectrum_avg = recon_spectrum_avg.cpu().numpy()
                    predicted_metrics_norm_avg = predicted_metrics_norm_avg.cpu().numpy()
                recon_spectrum_std = None
                predicted_metrics_std = None
            
            predicted_metrics_denorm_avg = denormalize_metrics(torch.tensor(predicted_metrics_norm_avg), dataset.metric_ranges).cpu().numpy()

            # 输出结果
            logger.info("Predicted Structural Parameters (Denormalized):")
            for i, params in enumerate(predicted_params_denorm):
                logger.info(f"Sample {i}: {', '.join([f'{name}={val:.4f}' for name, val in zip(dataset.param_names, params)])}")
            
            logger.info("Predicted Physical Metrics (Denormalized):")
            for i, metrics in enumerate(predicted_metrics_denorm_avg):
                logger.info(f"Sample {i}: {', '.join([f'{name}={val:.4f}' for name, val in zip(dataset.metric_names, metrics)])}")

            # 保存预测结果到CSV
            if args.output_prediction_path:
                output_df_columns = dataset.param_names + [f'recon_spec_point_{j}' for j in range(cfg.NUM_SPECTRUM_POINTS)] + dataset.metric_names
                output_data = np.hstack((predicted_params_denorm, recon_spectrum_avg, predicted_metrics_denorm_avg))
                
                import pandas as pd
                output_df = pd.DataFrame(output_data, columns=output_df_columns)
                
                # 如果有不确定性，也保存
                if args.mc_dropout_samples > 0:
                    for i, param_name in enumerate(dataset.param_names):
                        output_df[f'{param_name}_std'] = np.std(denormalize_params(mc_spec_samples[:, :, i], dataset.param_ranges), axis=0)
                    # 也可以保存光谱和指标的不确定性
                    # output_df[[f'recon_spec_point_{j}_std' for j in range(cfg.NUM_SPECTRUM_POINTS)]] = recon_spectrum_std
                    # output_df[[f'{name}_std' for name in dataset.metric_names]] = denormalize_metrics(torch.tensor(predicted_metrics_std), dataset.metric_ranges).cpu().numpy()

                output_df.to_csv(args.output_prediction_path, index=False)
                logger.info(f"Prediction results saved to {args.output_prediction_path}")

            # 绘制预测结果 (如果启用了)
            if args.plot_results:
                logger.info("\n--- Saving Prediction Plots ---")
                prediction_plot_dir = os.path.join(cfg.PROJECT_ROOT, "prediction_plots", f"{logger.experiment_log_dir.split(os.sep)[-1]}")
                os.makedirs(prediction_plot_dir, exist_ok=True)
                
                for i in range(new_spectrum_data.shape[0]):
                    plot_spectrum_comparison(new_spectrum_data[i], recon_spectrum_avg[i], dataset.frequencies, i, save_dir=prediction_plot_dir)
                    if args.mc_dropout_samples > 0:
                        plot_spectrum_uncertainty(new_spectrum_data[i], np.array(mc_spec_preds)[:, i, :], dataset.frequencies, i, save_dir=prediction_plot_dir)
                logger.info(f"Prediction plots saved to: {prediction_plot_dir}")


        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            exit(1)

        logger.info("--- Prediction Complete ---")

    logger.close() # 关闭 Logger
    print("--- Project Execution Complete ---")

if __name__ == '__main__':
    main()
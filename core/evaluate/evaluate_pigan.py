# PI_GAN_THZ/core/evaluate/evaluate_pigan.py

import sys
import os
import torch
import numpy as np
import argparse

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模型、数据加载器和绘图工具
from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.plot_utils import plot_losses, plot_generated_samples
from core.utils.set_seed import set_seed

def evaluate_pigan(num_samples_to_plot: int = 5):
    """
    评估和可视化 PI-GAN 的训练结果。
    Args:
        num_samples_to_plot (int): 要可视化生成样本的图片数量。
    """
    print("\n--- Starting PI-GAN Evaluation Script ---")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置随机种子，确保结果可复现
    set_seed(cfg.RANDOM_SEED)
    print(f"Random seed set to: {cfg.RANDOM_SEED}")

    # 确保创建必要的目录，包括 plots 目录
    cfg.create_directories() 

    # --- 数据加载 ---
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please check config.py and ensure the CSV file is there.")
        sys.exit(1)

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    print(f"Dataset size: {len(dataset)} samples")

    # --- 模型初始化和加载 ---
    generator = Generator(input_dim=cfg.SPECTRUM_DIM, output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM).to(device)
    discriminator = Discriminator(input_spec_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM, 
                                  input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM).to(device)
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    # 加载最终训练好的模型
    gen_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth")
    disc_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth")
    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth") # 或 forward_model_pretrained.pth

    if not os.path.exists(gen_model_path) or not os.path.exists(disc_model_path) or not os.path.exists(fwd_model_path):
        print(f"Error: One or more final PI-GAN models not found in {cfg.SAVED_MODELS_DIR}.")
        print("Please ensure PI-GAN training has completed and models are saved, or specify correct paths.")
        sys.exit(1)
    
    generator.load_state_dict(torch.load(gen_model_path, map_location=device))
    discriminator.load_state_dict(torch.load(disc_model_path, map_location=device))
    forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device)) # 加载前向模型权重

    generator.eval() # 切换到评估模式
    discriminator.eval() # 切换到评估模式
    forward_model.eval() # 切换到评估模式
    print(f"Loaded final Generator, Discriminator, and ForwardModel from {cfg.SAVED_MODELS_DIR}")

    # --- 加载损失历史 ---
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "pigan_loss_history.pt")
    if not os.path.exists(loss_history_path):
        print(f"Warning: PI-GAN training loss history not found at {loss_history_path}. Loss plots cannot be generated.")
        loss_history = {}
    else:
        loss_history = torch.load(loss_history_path)
        print(f"Loaded PI-GAN training loss history from {loss_history_path}")

    # --- 绘制损失曲线 ---
    if loss_history:
        epochs_for_plot = [(i + 1) * cfg.LOG_INTERVAL for i in range(len(loss_history['g_losses']))]
        
        print("\n--- Generating PI-GAN Training Loss Plots ---")
        # 绘制总损失
        plot_losses(
            epochs=epochs_for_plot,
            losses={
                'Generator Loss': loss_history['g_losses'],
                'Discriminator Loss': loss_history['d_losses']
            },
            title='Generator and Discriminator Losses over Epochs',
            xlabel='Epoch',
            ylabel='Loss',
            save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_gan_losses.png')
        )

        # 绘制生成器各子损失
        plot_losses(
            epochs=epochs_for_plot,
            losses={
                'Adversarial Loss': loss_history['adv_losses'],
                'Spectrum Reconstruction Loss': loss_history['recon_spec_losses'],
                'Metrics Reconstruction Loss': loss_history['recon_metrics_losses'],
                'Maxwell Loss': loss_history['maxwell_losses'],
                'LC Model Loss': loss_history['lc_losses'],
                'Param Range Loss': loss_history['param_range_losses'],
                'BNN KL Loss': loss_history['bnn_kl_losses']
            },
            title='Generator Sub-Losses over Epochs',
            xlabel='Epoch',
            ylabel='Loss',
            save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_generator_sub_losses.png')
        )
        print(f"Loss plots saved to {cfg.PLOTS_DIR}")
    else:
        print("No loss history available to plot.")

    # --- 生成样本可视化 ---
    print("\n--- Generating PI-GAN Sample Visualizations ---")
    with torch.no_grad():
        if num_samples_to_plot > len(dataset):
            num_samples_to_plot = len(dataset)
            print(f"Warning: num_samples_to_plot exceeds dataset size. Plotting all {num_samples_to_plot} samples.")
        
        # 随机选择一批真实光谱进行生成和可视化
        sample_indices = np.random.choice(len(dataset), num_samples_to_plot, replace=False)
        sample_real_spectrums = torch.stack([dataset[i][0] for i in sample_indices]).to(device)
        sample_real_params_denorm = torch.stack([dataset[i][1] for i in sample_indices]).to(device) # 真实未归一化参数

        # 通过生成器预测参数
        sample_predicted_params_norm = generator(sample_real_spectrums)
        sample_predicted_params_denorm = denormalize_params(sample_predicted_params_norm, dataset.param_ranges)

        # 通过前向模型重构光谱 (用于循环一致性检查)
        recon_spectrums, _ = forward_model(sample_predicted_params_norm)

        # 将张量移动回CPU并转换为NumPy以便绘图
        sample_real_spectrums_np = sample_real_spectrums.cpu().numpy()
        recon_spectrums_np = recon_spectrums.cpu().numpy()
        sample_real_params_denorm_np = sample_real_params_denorm.cpu().numpy()
        sample_predicted_params_denorm_np = sample_predicted_params_denorm.cpu().numpy()
        
        frequencies = dataset.frequencies # 从 MetamaterialDataset 中获取频率

        plot_generated_samples(
            real_spectrums=sample_real_spectrums_np,
            recon_spectrums=recon_spectrums_np,
            real_params=sample_real_params_denorm_np,
            predicted_params=sample_predicted_params_denorm_np,
            frequencies=frequencies,
            num_samples=num_samples_to_plot,
            save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_generated_samples.png')
        )
    print(f"PI-GAN sample plots saved to {cfg.PLOTS_DIR}")
    print("--- PI-GAN Evaluation Script Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained PI-GAN model.")
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to plot for visualization (default: 5)')
    args = parser.parse_args()
    
    evaluate_pigan(num_samples_to_plot=args.num_samples)
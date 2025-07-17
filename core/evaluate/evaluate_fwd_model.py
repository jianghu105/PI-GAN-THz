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
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_metrics
from core.utils.plot_utils import plot_losses, plot_fwd_model_predictions
from core.utils.set_seed import set_seed # 导入设置随机种子的函数

def evaluate_forward_model(num_samples_to_plot: int = 5):
    """
    评估和可视化预训练的前向模型。
    Args:
        num_samples_to_plot (int): 要可视化预测结果的样本数量。
    """
    print("\n--- Starting Forward Model Evaluation Script ---")
    sys.stdout.flush() # 立即刷新输出

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sys.stdout.flush()

    # 设置随机种子，确保结果可复现
    set_seed(cfg.RANDOM_SEED)
    print(f"Random seed set to: {cfg.RANDOM_SEED}")
    sys.stdout.flush()

    # 确保创建必要的目录，包括 plots 目录
    cfg.create_directories()
    print("All necessary directories ensured to exist.") # 英文化
    sys.stdout.flush()

    # --- 数据加载 ---
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please check config.py and ensure the CSV file is there.")
        sys.stdout.flush() # 立即刷新错误输出
        sys.exit(1)

    print(f"Loading data from {data_path}...") # 英文化
    sys.stdout.flush()

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    print(f"Dataset size: {len(dataset)} samples") # 英文化
    sys.stdout.flush()

    # --- 模型初始化和加载 ---
    forward_model = ForwardModel(
        input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
        output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
        output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM
    ).to(device)

    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth") 
    if not os.path.exists(fwd_model_path):
        print(f"Error: Pretrained Forward Model not found at {fwd_model_path}. Please run pretrain_fwd_model.py first, or ensure 'forward_model_final.pth' exists.")
        sys.stdout.flush() # 立即刷新错误输出
        sys.exit(1)
    
    forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device))
    forward_model.eval() # 切换到评估模式
    print(f"Loaded pretrained ForwardModel from {fwd_model_path}")
    sys.stdout.flush()

    # --- 加载损失历史 ---
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "fwd_pretrain_loss_history.pt")
    epoch_losses = [] # 默认初始化为空列表

    if not os.path.exists(loss_history_path):
        print(f"Warning: Forward Model pretraining loss history not found at {loss_history_path}. Loss plot cannot be generated.")
        sys.stdout.flush()
    else:
        try:
            loaded_history = torch.load(loss_history_path)
            # 根据您提供的文件内容，损失值存储在字典的 'train_losses' 键下
            if isinstance(loaded_history, dict) and 'train_losses' in loaded_history:
                epoch_losses = loaded_history['train_losses']
                print(f"Loaded Forward Model pretraining loss history from {loss_history_path} (key 'train_losses')")
            elif isinstance(loaded_history, list): # 如果直接是一个列表 (备用兼容)
                epoch_losses = loaded_history
                print(f"Loaded Forward Model pretraining loss history from {loss_history_path} (as a list)")
            else:
                print(f"Warning: Loss history file {loss_history_path} has an unexpected format. Cannot plot losses.")
                epoch_losses = [] # 重置为无法绘制的状态
            sys.stdout.flush()
        except Exception as e:
            print(f"Error loading loss history from {loss_history_path}: {e}. Cannot plot losses.")
            sys.stdout.flush()
            epoch_losses = [] # 重置为无法绘制的状态

    # --- 绘制损失曲线 ---
    if epoch_losses:
        print("\n--- Generating Pretraining Loss Plots ---")
        sys.stdout.flush()
        plot_losses(
            epochs=list(range(1, len(epoch_losses) + 1)),
            losses={'Forward Model Loss': epoch_losses},
            title='Forward Model Pretraining Loss over Epochs',
            xlabel='Epoch',
            ylabel='MSE Loss',
            save_path=os.path.join(cfg.PLOTS_DIR, 'fwd_pretrain_loss.png')
        )
        print(f"Loss plot saved to {os.path.join(cfg.PLOTS_DIR, 'fwd_pretrain_loss.png')}")
        sys.stdout.flush()
    else:
        print("No loss history available to plot.")
        sys.stdout.flush()

    # --- 生成样本可视化 ---
    print("\n--- Generating Forward Model Prediction Samples ---")
    sys.stdout.flush()
    with torch.no_grad():
        # 从数据集中随机选择一些样本进行可视化
        if num_samples_to_plot <= 0: # 增加对 <= 0 样本数的处理
            print("Warning: Number of samples to plot is 0 or negative, skipping sample visualization.")
            sys.stdout.flush()
            return # 提前退出
        
        if num_samples_to_plot > len(dataset):
            num_samples_to_plot = len(dataset)
            print(f"Warning: num_samples_to_plot exceeds dataset size. Plotting all {num_samples_to_plot} samples.")
            sys.stdout.flush()
        
        if num_samples_to_plot == 0: # 再次检查调整后的样本数
            print("Not enough samples available for generating visualization.")
            sys.stdout.flush()
            return # 提前退出

        sample_indices = np.random.choice(len(dataset), num_samples_to_plot, replace=False)
        
        # 获取真实数据
        sample_real_params_norm = torch.stack([dataset[i][2] for i in sample_indices]).to(device) # 归一化的真实参数
        sample_real_spectrums = torch.stack([dataset[i][0] for i in sample_indices]).to(device) # 真实光谱
        sample_real_metrics_norm = torch.stack([dataset[i][4] for i in sample_indices]).to(device) # 归一化的真实指标

        # 通过前向模型预测光谱和指标
        predicted_spectrums, predicted_metrics_norm = forward_model(sample_real_params_norm)

        # 反归一化预测的指标和真实指标，用于可视化
        real_metrics_denorm_for_plot = denormalize_metrics(sample_real_metrics_norm, dataset.metric_ranges).cpu().numpy()
        predicted_metrics_denorm_for_plot = denormalize_metrics(predicted_metrics_norm, dataset.metric_ranges).cpu().numpy()

        # 将张量移动回 CPU 并转换为 NumPy 数组以便绘图
        sample_real_params_denorm_np = np.stack([dataset[i][1] for i in sample_indices]) # 获取非归一化的真实参数用于绘图
        sample_real_spectrums_np = sample_real_spectrums.cpu().numpy()
        predicted_spectrums_np = predicted_spectrums.cpu().numpy()
        
        frequencies = dataset.frequencies # 从 MetamaterialDataset 获取频率数据

        plot_fwd_model_predictions(
            real_params=sample_real_params_denorm_np,
            real_spectrums=sample_real_spectrums_np,
            predicted_spectrums=predicted_spectrums_np,
            real_metrics=real_metrics_denorm_for_plot,
            predicted_metrics=predicted_metrics_denorm_for_plot,
            frequencies=frequencies,
            num_samples=num_samples_to_plot,
            save_path=os.path.join(cfg.PLOTS_DIR, 'fwd_model_predictions.png'),
            metric_names=dataset.metric_names # 修正为 metric_names
        )
    print(f"Forward model prediction plots saved to {os.path.join(cfg.PLOTS_DIR, 'fwd_model_predictions.png')}")
    sys.stdout.flush()
    print("--- Forward Model Evaluation Script Finished ---")
    sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the pretrained Forward Model.")
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to plot for visualization (default: 5)')
    args = parser.parse_args()
    
    evaluate_forward_model(num_samples_to_plot=args.num_samples)
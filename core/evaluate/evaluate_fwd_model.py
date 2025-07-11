# PI_GAN_THZ/core/evaluate/evaluate_fwd_model.py

import sys
import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader # 确保导入 DataLoader
from tqdm.notebook import tqdm # <<<<<<< 更改为 tqdm.notebook

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模型、数据加载器和绘图工具
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.plot_utils import plot_losses, plot_fwd_model_predictions
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_mse # <<<<<<< 确保导入 criterion_mse

def evaluate_fwd_model(num_samples_to_plot: int = 5):
    """
    评估和可视化预训练的前向模型。
    Args:
        num_samples_to_plot (int): 要可视化预测结果的图片数量。
    """
    print("\n--- 正在启动前向模型评估脚本 ---")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 设置随机种子，确保结果可复现
    set_seed(cfg.RANDOM_SEED)
    print(f"随机种子已设置为: {cfg.RANDOM_SEED}")

    # 确保创建必要的目录，包括 plots 目录
    cfg.create_directories()
    print(f"图像将保存到: {cfg.PLOTS_DIR}")

    # --- 数据加载 ---
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        print(f"错误: 数据集未找到，路径为 {data_path}。请检查 config.py 并确保 CSV 文件存在。")
        # 在Colab中，当用!python运行时，sys.exit()会直接停止cell执行，可能看不到错误。
        # 这里为了调试，先不直接退出，而是打印错误并尝试继续。
        return # 如果数据丢失，优雅地返回

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.BATCH_SIZE, # 评估时使用 config 中的 batch_size
        shuffle=False, # 评估无需打乱数据
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    print(f"数据集已加载，包含 {len(dataset)} 个样本。")
    print(f"评估批次数量: {len(dataloader)}")

    # --- 模型初始化和加载 ---
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_pretrained.pth")
    if not os.path.exists(model_path):
        print(f"错误: 预训练的前向模型未找到，路径为 {model_path}。")
        print("请确保 'pretrain_fwd_model.py' 已经成功运行并保存了模型。")
        return # 如果模型丢失，优雅地返回
    
    forward_model.load_state_dict(torch.load(model_path, map_location=device))
    forward_model.eval() # 将模型设置为评估模式
    print(f"已从 {model_path} 加载预训练的前向模型。")

    # --- 加载损失历史 ---
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "fwd_pretrain_loss_history.pt")
    if not os.path.exists(loss_history_path):
        print(f"警告: 前向模型预训练损失历史未找到，路径为 {loss_history_path}。无法生成损失图。")
        loss_history = {}
    else:
        # 假设 fwd_pretrain_loss_history.pt 保存的是一个字典 {'train_losses': [...] }
        loaded_history = torch.load(loss_history_path)
        # 确保它是一个字典，并且包含 'train_losses' 键
        if isinstance(loaded_history, dict) and 'train_losses' in loaded_history:
            loss_history = loaded_history
        else: # 如果不是预期格式，则视为无效
            print(f"警告: 损失历史文件 {loss_history_path} 格式不正确。无法生成损失图。")
            loss_history = {}
        print(f"已从 {loss_history_path} 加载前向模型训练损失历史。")

    # --- 绘制损失曲线 ---
    if loss_history and 'train_losses' in loss_history and loss_history['train_losses']: # 确保损失历史和 specific key 存在且不为空
        epochs_for_plot = [(i + 1) * cfg.LOG_INTERVAL for i in range(len(loss_history['train_losses']))]
        # 如果LOG_INTERVAL不用于fwd pretrain的epoch logging，则直接用1作为步长
        if len(epochs_for_plot) != len(loss_history['train_losses']): # 如果LOG_INTERVAL不匹配实际的epoch数量
             epochs_for_plot = list(range(1, len(loss_history['train_losses']) + 1))
        
        print("\n--- 正在生成前向模型损失图 ---")
        plot_losses(
            epochs=epochs_for_plot,
            losses={'训练损失': loss_history['train_losses']},
            title='前向模型预训练损失随 Epoch 变化',
            xlabel='Epoch',
            ylabel='损失',
            save_path=os.path.join(cfg.PLOTS_DIR, 'fwd_model_train_loss.png')
        )
        print(f"损失图已保存到 {cfg.PLOTS_DIR}")
    else:
        print("没有可用于绘制前向模型的有效损失历史。")

    # --- 进行预测并可视化 ---
    print("\n--- 正在生成前向模型预测可视化 ---")
    mse_criterion = criterion_mse() # 初始化 MSE 准则用于定量评估

    all_real_params = []
    all_real_spectrums = []
    all_predicted_spectrums = []
    all_real_metrics = []
    all_predicted_metrics = []
    
    total_spectrum_mse = 0.0
    total_metrics_mse = 0.0
    num_batches = 0

    with torch.no_grad():
        # 使用 tqdm 包装 dataloader，以显示评估进度
        for i, (real_spectrum, real_params_denorm, real_params_norm, real_metrics_denorm, real_metrics_norm) in tqdm(enumerate(dataloader), desc="正在评估前向模型", leave=False):
            real_params_norm = real_params_norm.to(device)
            real_spectrum = real_spectrum.to(device)
            real_metrics_norm = real_metrics_norm.to(device)

            predicted_spectrum, predicted_metrics_norm = forward_model(real_params_norm)

            # 计算 MSE
            total_spectrum_mse += mse_criterion(predicted_spectrum, real_spectrum).item()
            total_metrics_mse += mse_criterion(predicted_metrics_norm, real_metrics_norm).item()
            num_batches += 1

            # 存储用于绘图 (只存储少量样本)
            if len(all_real_params) < num_samples_to_plot:
                batch_to_take = min(num_samples_to_plot - len(all_real_params), real_params_norm.size(0))
                all_real_params.append(real_params_denorm[:batch_to_take].cpu().numpy())
                all_real_spectrums.append(real_spectrum[:batch_to_take].cpu().numpy())
                all_predicted_spectrums.append(predicted_spectrum[:batch_to_take].cpu().numpy())
                all_real_metrics.append(real_metrics_denorm[:batch_to_take].cpu().numpy()) # 使用去归一化的指标用于绘图
                all_predicted_metrics.append(denormalize_metrics(predicted_metrics_norm[:batch_to_take], dataset.metric_ranges).cpu().numpy())

        # 连接样本用于绘图
        if all_real_params:
            real_params_np = np.concatenate(all_real_params, axis=0)[:num_samples_to_plot]
            real_spectrums_np = np.concatenate(all_real_spectrums, axis=0)[:num_samples_to_plot]
            predicted_spectrums_np = np.concatenate(all_predicted_spectrums, axis=0)[:num_samples_to_plot]
            real_metrics_np = np.concatenate(all_real_metrics, axis=0)[:num_samples_to_plot]
            predicted_metrics_np = np.concatenate(all_predicted_metrics, axis=0)[:num_samples_to_plot]

            frequencies = dataset.frequencies # 从数据集中获取频率
            metric_names = dataset.metric_names # 从数据集中获取指标名称

            plot_fwd_model_predictions(
                real_params=real_params_np,
                real_spectrums=real_spectrums_np,
                predicted_spectrums=predicted_spectrums_np,
                real_metrics=real_metrics_np,
                predicted_metrics=predicted_metrics_np,
                frequencies=frequencies,
                num_samples=num_samples_to_plot,
                save_path=os.path.join(cfg.PLOTS_DIR, 'fwd_model_predictions.png'),
                metric_names=metric_names
            )
            print(f"预测图已保存到 {cfg.PLOTS_DIR}")
        else:
            print("没有处理样本用于绘图。请检查数据加载器和数据集。")

    if num_batches > 0:
        avg_spectrum_mse = total_spectrum_mse / num_batches
        avg_metrics_mse = total_metrics_mse / num_batches
        print(f"\n平均光谱 MSE: {avg_spectrum_mse:.4f}")
        print(f"平均指标 MSE: {avg_metrics_mse:.4f}")
    else:
        print("没有处理批次用于 MSE 计算。")

    print("--- 前向模型评估脚本已完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估已训练的前向模型。")
    parser.add_argument('--num_samples', type=int, default=5,
                        help='要可视化绘制的样本数量 (默认: 5)')
    args = parser.parse_args()
    
    evaluate_fwd_model(num_samples_to_plot=args.num_samples)
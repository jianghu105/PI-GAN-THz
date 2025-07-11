# PI_GAN_THZ/core/utils/plot_utils.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_losses(epochs: list, losses: dict, title: str, xlabel: str, ylabel: str, save_path: str):
    """
    绘制损失曲线图。

    Args:
        epochs (list): 对应损失值的 epoch 列表。
        losses (dict): 包含多个损失曲线的字典，键是曲线名称，值是损失值列表。
                       例如：{'Loss A': [val1, val2, ...], 'Loss B': [val1, val2, ...]}
        title (str): 图表标题。
        xlabel (str): X轴标签。
        ylabel (str): Y轴标签。
        save_path (str): 保存图表的路径。
    """
    plt.figure(figsize=(10, 6))
    for label, data in losses.items():
        plt.plot(epochs, data, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: {save_path}")

def plot_generated_samples(
    real_spectrums: np.ndarray, 
    recon_spectrums: np.ndarray, 
    real_params: np.ndarray, 
    predicted_params: np.ndarray, 
    frequencies: np.ndarray,
    num_samples: int = 5, 
    save_path: str = 'generated_samples.png'
):
    """
    可视化真实光谱与重构光谱，以及对应的真实参数与预测参数（用于 PI-GAN 的生成结果）。

    Args:
        real_spectrums (np.ndarray): 真实光谱数据 (batch_size, spectrum_dim)。
        recon_spectrums (np.ndarray): 重构光谱数据 (batch_size, spectrum_dim)。
        real_params (np.ndarray): 真实结构参数 (batch_size, param_dim)。
        predicted_params (np.ndarray): 预测结构参数 (batch_size, param_dim)。
        frequencies (np.ndarray): 光谱对应的频率点 (spectrum_dim,)。
        num_samples (int): 要可视化的样本数量。
        save_path (str): 保存图表的路径。
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, num_samples * 4)) # 两列：光谱对比和参数对比

    if num_samples == 1: # 如果只有一个样本，axes 可能不是二维数组，做一下处理
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # 第一列：光谱对比
        ax1 = axes[i, 0]
        ax1.plot(frequencies, real_spectrums[i], label='Real Spectrum', color='blue')
        ax1.plot(frequencies, recon_spectrums[i], label='Reconstructed Spectrum', color='red', linestyle='--')
        ax1.set_title(f'Sample {i+1}: Spectrum Comparison')
        ax1.set_xlabel('Frequency')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True)

        # 第二列：参数对比
        ax2 = axes[i, 1]
        param_indices = np.arange(real_params.shape[1])
        ax2.bar(param_indices - 0.2, real_params[i], width=0.4, label='Real Params', color='skyblue')
        ax2.bar(param_indices + 0.2, predicted_params[i], width=0.4, label='Predicted Params', color='lightcoral')
        ax2.set_xticks(param_indices)
        ax2.set_xticklabels([f'P{j+1}' for j in param_indices]) 
        ax2.set_title(f'Sample {i+1}: Parameter Comparison')
        ax2.set_xlabel('Parameter Index')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Generated sample plots saved to: {save_path}")

def plot_fwd_model_predictions(
    real_params: np.ndarray,
    real_spectrums: np.ndarray,
    predicted_spectrums: np.ndarray,
    real_metrics: np.ndarray,
    predicted_metrics: np.ndarray,
    frequencies: np.ndarray,
    num_samples: int = 5,
    save_path: str = 'fwd_model_predictions.png',
    metric_names: list = None
):
    """
    可视化前向模型预测的 Spectra 和 Metrics。

    Args:
        real_params (np.ndarray): 真实结构参数 (batch_size, param_dim)。
        real_spectrums (np.ndarray): 真实光谱数据 (batch_size, spectrum_dim)。
        predicted_spectrums (np.ndarray): 预测光谱数据 (batch_size, spectrum_dim)。
        real_metrics (np.ndarray): 真实物理指标数据 (batch_size, metrics_dim)。
        predicted_metrics (np.ndarray): 预测物理指标数据 (batch_size, metrics_dim)。
        frequencies (np.ndarray): 光谱对应的频率点 (spectrum_dim,)。
        num_samples (int): 要可视化的样本数量。
        save_path (str): 保存图表的路径。
        metric_names (list): 物理指标的名称列表，用于图例。
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(18, num_samples * 4)) # 三列：参数、光谱对比、指标对比

    if num_samples == 1: # 如果只有一个样本，axes 可能不是二维数组，做一下处理
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_samples):
        # 第一列：输入参数
        ax1 = axes[i, 0]
        param_indices = np.arange(real_params.shape[1])
        ax1.bar(param_indices, real_params[i], color='lightgray')
        ax1.set_xticks(param_indices)
        ax1.set_xticklabels([f'P{j+1}' for j in param_indices])
        ax1.set_title(f'Sample {i+1}: Input Parameters')
        ax1.set_ylabel('Value')
        ax1.grid(True, axis='y')

        # 第二列：光谱对比
        ax2 = axes[i, 1]
        ax2.plot(frequencies, real_spectrums[i], label='Real Spectrum', color='blue')
        ax2.plot(frequencies, predicted_spectrums[i], label='Predicted Spectrum', color='red', linestyle='--')
        ax2.set_title(f'Sample {i+1}: Spectrum Prediction')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True)

        # 第三列：指标对比
        ax3 = axes[i, 2]
        metric_indices = np.arange(real_metrics.shape[1])
        width = 0.35
        ax3.bar(metric_indices - width/2, real_metrics[i], width, label='Real Metrics', color='skyblue')
        ax3.bar(metric_indices + width/2, predicted_metrics[i], width, label='Predicted Metrics', color='lightcoral')
        ax3.set_xticks(metric_indices)
        ax3.set_xticklabels(metric_names if metric_names else [f'M{j+1}' for j in metric_indices], rotation=45, ha='right')
        ax3.set_title(f'Sample {i+1}: Metrics Prediction')
        ax3.set_xlabel('Metric')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Prediction plots saved to: {save_path}")
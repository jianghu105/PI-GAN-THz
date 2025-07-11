# PI_GAN_THZ/core/utils/visual.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_spectrum_comparison(real_spectrum: np.ndarray, predicted_spectrum: np.ndarray, 
                             frequencies: np.ndarray, sample_idx: int, save_dir: str = None):
    """
    绘制真实光谱与重建光谱的对比图。

    Args:
        real_spectrum (np.ndarray): 真实光谱数据 (单个样本或批次)。
        predicted_spectrum (np.ndarray): 重建光谱数据 (单个样本或批次)。
        frequencies (np.ndarray): 对应的频率点。
        sample_idx (int): 用于标识样本的索引。
        save_dir (str, optional): 保存图片的目录。如果为 None，则显示图片。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, real_spectrum, label='Real Spectrum', color='blue')
    plt.plot(frequencies, predicted_spectrum, label='Reconstructed Spectrum', color='red', linestyle='--')
    plt.title(f'Spectrum Comparison - Sample {sample_idx}')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Transmission/Absorption')
    plt.legend()
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'spectrum_comparison_sample_{sample_idx}.png'))
        plt.close()
    else:
        plt.show()

def plot_parameter_prediction(real_params: np.ndarray, predicted_params: np.ndarray, 
                              param_names: list, save_dir: str = None):
    """
    绘制真实参数与预测参数的散点图。

    Args:
        real_params (np.ndarray): 真实参数数据 (所有样本)。
        predicted_params (np.ndarray): 预测参数数据 (所有样本)。
        param_names (list): 参数名称列表，用于x轴标签。
        save_dir (str, optional): 保存图片的目录。如果为 None，则显示图片。
    """
    num_params = real_params.shape[1]
    fig, axes = plt.subplots(1, num_params, figsize=(5 * num_params, 5))
    if num_params == 1: # 如果只有一个参数，axes不是数组
        axes = [axes] 

    for i in range(num_params):
        axes[i].scatter(real_params[:, i], predicted_params[:, i], alpha=0.5)
        axes[i].plot([min(real_params[:, i]), max(real_params[:, i])],
                     [min(real_params[:, i]), max(real_params[:, i])],
                     'r--', label='Ideal') # 添加对角线表示理想情况
        axes[i].set_title(f'{param_names[i]} Prediction')
        axes[i].set_xlabel(f'Real {param_names[i]}')
        axes[i].set_ylabel(f'Predicted {param_names[i]}')
        axes[i].legend()
        axes[i].grid(True)
    
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'parameter_predictions.png'))
        plt.close()
    else:
        plt.show()

def plot_metrics_prediction(real_metrics: np.ndarray, predicted_metrics: np.ndarray, 
                            metric_names: list, save_dir: str = None):
    """
    绘制真实物理指标与预测物理指标的散点图。

    Args:
        real_metrics (np.ndarray): 真实物理指标数据 (所有样本)。
        predicted_metrics (np.ndarray): 预测物理指标数据 (所有样本)。
        metric_names (list): 物理指标名称列表。
        save_dir (str, optional): 保存图片的目录。如果为 None，则显示图片。
    """
    num_metrics = real_metrics.shape[1]
    # 计算子图布局，例如 2行 X N列
    cols = 4
    rows = (num_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = axes.flatten() # 将 axes 展平，方便迭代

    for i in range(num_metrics):
        axes[i].scatter(real_metrics[:, i], predicted_metrics[:, i], alpha=0.5)
        axes[i].plot([min(real_metrics[:, i]), max(real_metrics[:, i])],
                     [min(real_metrics[:, i]), max(real_metrics[:, i])],
                     'r--', label='Ideal')
        axes[i].set_title(f'{metric_names[i]} Prediction')
        axes[i].set_xlabel(f'Real {metric_names[i]}')
        axes[i].set_ylabel(f'Predicted {metric_names[i]}')
        axes[i].legend()
        axes[i].grid(True)
    
    # 隐藏多余的子图
    for j in range(num_metrics, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'metrics_predictions.png'))
        plt.close()
    else:
        plt.show()

def plot_spectrum_uncertainty(real_spectrum: np.ndarray, mc_predicted_spectrum_samples: np.ndarray,
                              frequencies: np.ndarray, sample_idx: int, save_dir: str = None):
    """
    绘制真实光谱和MC Dropout预测光谱的均值及不确定性带。

    Args:
        real_spectrum (np.ndarray): 真实光谱数据 (单个样本)。
        mc_predicted_spectrum_samples (np.ndarray): MC Dropout采样的预测光谱 (mc_samples, spectrum_dim)。
        frequencies (np.ndarray): 对应的频率点。
        sample_idx (int): 用于标识样本的索引。
        save_dir (str, optional): 保存图片的目录。
    """
    mean_spectrum = np.mean(mc_predicted_spectrum_samples, axis=0)
    std_spectrum = np.std(mc_predicted_spectrum_samples, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, real_spectrum, label='Real Spectrum', color='blue', linewidth=2)
    plt.plot(frequencies, mean_spectrum, label='MC Mean Prediction', color='red', linestyle='--', linewidth=2)
    plt.fill_between(frequencies, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                     color='red', alpha=0.2, label='MC Uncertainty (±1 Std Dev)')
    
    plt.title(f'Spectrum Prediction with Uncertainty - Sample {sample_idx}')
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Transmission/Absorption')
    plt.legend()
    plt.grid(True)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f'spectrum_uncertainty_sample_{sample_idx}.png'))
        plt.close()
    else:
        plt.show()
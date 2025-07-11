# PI_GAN_THZ/core/utils/plot_utils.py
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.font_manager as fm # 确保导入字体管理器

# --- Matplotlib 中文及显示配置 (在 plot_utils 中配置，以便所有绘图函数都能应用) ---
# 注意：在 Colab 中，还需要在 Notebook 开头单独运行安装字体和清除缓存的单元格
# 这里只是设置参数，确保绘图时使用正确的字体
try:
    # 尝试设置中文字体，优先使用 Noto Sans CJK SC
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'SimHei', 'Arial Unicode MS', 'sans-serif']
except:
    # 如果 Colab 环境没有这些字体，可以使用通用无衬线字体，但中文可能仍是方框
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 确保在 Colab 中显示图像的魔术命令在 Notebook 顶部，而不是这里
# %matplotlib inline

# --------------------------------------------------------------------------

def plot_losses(epochs: list, losses: dict, title: str, xlabel: str, ylabel: str, save_path: str):
    """
    绘制损失曲线。

    Args:
        epochs (list): x轴数据，通常是 epoch 或 iteration。
        losses (dict): 包含损失名称和损失值列表的字典，例如 {'训练损失': [l1, l2, ...]}。
        title (str): 图表标题。
        xlabel (str): x轴标签。
        ylabel (str): y轴标签。
        save_path (str): 图表保存路径。
    """
    plt.figure(figsize=(10, 6))
    for label, loss_values in losses.items():
        plt.plot(epochs, loss_values, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show() # <<<<<<< 关键：显示图表
    plt.close() # 关闭图表，释放内存


def plot_fwd_model_predictions(real_params: np.ndarray, real_spectrums: np.ndarray, 
                               predicted_spectrums: np.ndarray, real_metrics: np.ndarray, 
                               predicted_metrics: np.ndarray, frequencies: np.ndarray, 
                               num_samples: int, save_path: str, metric_names: list):
    """
    可视化前向模型的预测结果，包括光谱和物理指标。

    Args:
        real_params (np.ndarray): 真实结构参数 (去归一化)。
        real_spectrums (np.ndarray): 真实光谱。
        predicted_spectrums (np.ndarray): 预测光谱。
        real_metrics (np.ndarray): 真实物理指标 (去归一化)。
        predicted_metrics (np.ndarray): 预测物理指标 (去归一化)。
        frequencies (np.ndarray): 频率范围。
        num_samples (int): 要绘制的样本数量。
        save_path (str): 保存图表的路径。
        metric_names (list): 物理指标的名称列表。
    """
    fig, axes = plt.subplots(num_samples, 2, figsize=(15, num_samples * 4)) # 两列：光谱和指标

    for i in range(num_samples):
        # 绘制光谱
        ax1 = axes[i, 0] if num_samples > 1 else axes[0]
        ax1.plot(frequencies, real_spectrums[i], label='真实光谱', linestyle='-', marker='o', markersize=2)
        ax1.plot(frequencies, predicted_spectrums[i], label='预测光谱', linestyle='--', marker='x', markersize=2)
        ax1.set_title(f'样本 {i+1} 光谱预测 (参数: {real_params[i,0]:.2f}, {real_params[i,1]:.2f})')
        ax1.set_xlabel('频率 (THz)')
        ax1.set_ylabel('透射率')
        ax1.legend()
        ax1.grid(True)

        # 绘制物理指标
        ax2 = axes[i, 1] if num_samples > 1 else axes[1]
        
        # 将参数和指标组合成字典，便于传递给表格
        data = []
        for j, name in enumerate(metric_names):
            data.append([name, f"{real_metrics[i, j]:.4f}", f"{predicted_metrics[i, j]:.4f}"])

        # 为指标创建一个表格
        table = ax2.table(cellText=data,
                          colLabels=['指标名称', '真实值', '预测值'],
                          cellLoc='center',
                          loc='center') # 将表格放置在轴的中心
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2) # 调整表格大小
        ax2.set_title(f'样本 {i+1} 物理指标预测')
        ax2.axis('off') # 关闭坐标轴

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show() # <<<<<<< 关键：显示图表
    plt.close() # 关闭图表，释放内存

# ... 其他绘图函数，如果存在，也需要添加 plt.show() 和 plt.close()
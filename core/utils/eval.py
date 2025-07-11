# PI_GAN_THZ/core/utils/eval.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 导入数据处理工具
from core.utils.data_loader import denormalize_params, denormalize_metrics

def load_model_state(model: nn.Module, model_path: str, device: torch.device):
    """
    加载模型的状态字典。

    Args:
        model (nn.Module): 待加载权重的模型实例。
        model_path (str): 模型状态字典的路径 (.pth 文件)。
        device (torch.device): 模型加载到的设备 (CPU/GPU)。
    Returns:
        nn.Module: 加载了权重的模型实例。
    """
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # 设置为评估模式
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        raise
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        raise
    return model

def evaluate_pigan_inverse_design(
    generator: nn.Module,
    forward_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    dataset_obj, # 传递整个数据集对象，以便访问参数范围和指标名称
    mc_dropout_samples: int = 0 # 进行MC Dropout采样次数，0表示不进行
):
    """
    评估 PIGAN 的逆向设计性能。

    Args:
        generator (nn.Module): 已训练的生成器模型。
        forward_model (nn.Module): 已训练的前向模型。
        dataloader (DataLoader): 用于评估的数据加载器 (通常是测试集)。
        device (torch.device): 运行评估的设备。
        dataset_obj: MetamaterialDataset 实例，用于访问参数归一化范围和指标名称。
        mc_dropout_samples (int): 如果 > 0，则执行 Monte Carlo Dropout 采样。
    Returns:
        dict: 包含各种评估指标的字典。
    """
    generator.eval()
    forward_model.eval() # 评估时保持评估模式

    all_real_spectrum = []
    all_predicted_spectrum = []
    all_real_params_denorm = []
    all_predicted_params_denorm = []
    all_real_metrics_norm = []
    all_predicted_metrics_norm = []

    # 用于MC Dropout的不确定性估计
    all_mc_predicted_spectrum = []
    all_mc_predicted_metrics = []

    with torch.no_grad():
        for i, (real_spectrum, real_params_denorm, real_params_norm, real_metrics_denorm, real_metrics_norm) in enumerate(dataloader):
            real_spectrum = real_spectrum.to(device)
            real_params_denorm = real_params_denorm.cpu().numpy() # 收集到CPU
            real_params_norm = real_params_norm.to(device)
            real_metrics_norm = real_metrics_norm.to(device)

            # --- 1. 生成器逆向设计 ---
            predicted_params_norm = generator(real_spectrum)
            predicted_params_denorm = denormalize_params(predicted_params_norm, dataset_obj.param_ranges).cpu().numpy()

            # --- 2. 前向模型重建光谱和指标 ---
            if mc_dropout_samples > 0:
                # 开启ForwardModel的Dropout层 (仅在评估时用于MC Dropout)
                for m in forward_model.modules():
                    if isinstance(m, nn.Dropout):
                        m.train() # 临时设置为训练模式以启用dropout

                mc_spec_preds = []
                mc_metrics_preds = []
                for _ in range(mc_dropout_samples):
                    recon_spectrum_mc, predicted_metrics_norm_mc = forward_model(predicted_params_norm)
                    mc_spec_preds.append(recon_spectrum_mc.cpu().numpy())
                    mc_metrics_preds.append(predicted_metrics_norm_mc.cpu().numpy())
                
                # 关闭ForwardModel的Dropout层
                for m in forward_model.modules():
                    if isinstance(m, nn.Dropout):
                        m.eval() # 恢复评估模式

                all_mc_predicted_spectrum.append(np.array(mc_spec_preds)) # (mc_samples, batch_size, spectrum_dim)
                all_mc_predicted_metrics.append(np.array(mc_metrics_preds)) # (mc_samples, batch_size, metrics_dim)

                # 使用MC平均值作为最终预测
                recon_spectrum = torch.tensor(np.mean(np.array(mc_spec_preds), axis=0)).to(device)
                predicted_metrics_norm = torch.tensor(np.mean(np.array(mc_metrics_preds), axis=0)).to(device)
            else:
                recon_spectrum, predicted_metrics_norm = forward_model(predicted_params_norm)

            all_real_spectrum.append(real_spectrum.cpu().numpy())
            all_predicted_spectrum.append(recon_spectrum.cpu().numpy())
            all_real_params_denorm.append(real_params_denorm)
            all_predicted_params_denorm.append(predicted_params_denorm)
            all_real_metrics_norm.append(real_metrics_norm.cpu().numpy())
            all_predicted_metrics_norm.append(predicted_metrics_norm.cpu().numpy())

    # 将列表转换为 numpy 数组
    real_spectrum_arr = np.concatenate(all_real_spectrum, axis=0)
    predicted_spectrum_arr = np.concatenate(all_predicted_spectrum, axis=0)
    real_params_denorm_arr = np.concatenate(all_real_params_denorm, axis=0)
    predicted_params_denorm_arr = np.concatenate(all_predicted_params_denorm, axis=0)
    real_metrics_norm_arr = np.concatenate(all_real_metrics_norm, axis=0)
    predicted_metrics_norm_arr = np.concatenate(all_predicted_metrics_norm, axis=0)

    # --- 3. 计算评估指标 ---
    metrics = {}

    # 光谱重建误差
    metrics['spectrum_mse'] = mean_squared_error(real_spectrum_arr, predicted_spectrum_arr)
    metrics['spectrum_mae'] = mean_absolute_error(real_spectrum_arr, predicted_spectrum_arr)
    metrics['spectrum_r2'] = r2_score(real_spectrum_arr, predicted_spectrum_arr)

    # 参数预测误差 (针对每个参数)
    metrics['param_mse'] = mean_squared_error(real_params_denorm_arr, predicted_params_denorm_arr)
    metrics['param_mae'] = mean_absolute_error(real_params_denorm_arr, predicted_params_denorm_arr)
    metrics['param_r2'] = r2_score(real_params_denorm_arr, predicted_params_denorm_arr)

    # 物理指标预测误差 (针对每个指标)
    metrics['metrics_mse'] = mean_squared_error(real_metrics_norm_arr, predicted_metrics_norm_arr)
    metrics['metrics_mae'] = mean_absolute_error(real_metrics_norm_arr, predicted_metrics_norm_arr)
    metrics['metrics_r2'] = r2_score(real_metrics_norm_arr, predicted_metrics_norm_arr)

    print("\n--- Evaluation Results ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # 如果进行了MC Dropout，返回原始的MC采样结果以便可视化不确定性
    if mc_dropout_samples > 0:
        mc_spectrum_preds_arr = np.concatenate(all_mc_predicted_spectrum, axis=1) # (mc_samples, total_samples, spectrum_dim)
        mc_metrics_preds_arr = np.concatenate(all_mc_predicted_metrics, axis=1) # (mc_samples, total_samples, metrics_dim)
        return metrics, real_spectrum_arr, predicted_spectrum_arr, \
               real_params_denorm_arr, predicted_params_denorm_arr, \
               real_metrics_norm_arr, predicted_metrics_norm_arr, \
               mc_spectrum_preds_arr, mc_metrics_preds_arr
    else:
        return metrics, real_spectrum_arr, predicted_spectrum_arr, \
               real_params_denorm_arr, predicted_params_denorm_arr, \
               real_metrics_norm_arr, predicted_metrics_norm_arr, \
               None, None
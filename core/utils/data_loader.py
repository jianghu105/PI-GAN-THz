# PI_GAN_THZ/core/utils/data_loader.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
import os

# --- 辅助函数：计算Q参数、FoM参数、灵敏度S (与之前相同，略) ---
def calculate_peak_parameters(frequency, transmission_db, peak_idx, baseline_transmission=0):
    f_res = frequency[peak_idx]
    t_min = transmission_db[peak_idx]
    half_depth_val = t_min + (baseline_transmission - t_min) / 2
    f_lower, f_upper = None, None
    for i in range(peak_idx - 1, -1, -1):
        if (transmission_db[i] >= half_depth_val and transmission_db[i+1] < half_depth_val) or \
           (transmission_db[i] < half_depth_val and transmission_db[i+1] >= half_depth_val):
            f_lower = frequency[i] + (half_depth_val - transmission_db[i]) * \
                      (frequency[i+1] - frequency[i]) / (transmission_db[i+1] - transmission_db[i])
            break
    for i in range(peak_idx + 1, len(frequency)):
        if (transmission_db[i] <= half_depth_val and transmission_db[i+1] > half_depth_val) or \
           (transmission_db[i] > half_depth_val and transmission_db[i+1] <= half_depth_val):
            f_upper = frequency[i] + (half_depth_val - transmission_db[i]) * \
                      (frequency[i+1] - frequency[i]) / (transmission_db[i+1] - transmission_db[i])
            break
    Q = np.nan
    FoM = np.nan
    if f_lower is not None and f_upper is not None and f_upper > f_lower:
        delta_f = f_upper - f_lower
        Q = f_res / delta_f
        # 修正FoM的计算，确保分母不为零且有意义
        FoM = Q / abs(t_min) if abs(t_min) > 1e-6 else np.nan # 使用一个小的 epsilon
    return f_res, Q, FoM

# --- 数据生成函数 (不再使用，但保留以防万一或作为参考) ---
def generate_single_terahertz_spectrum_and_params(frequency, r1, r2, w, g, apply_offset=True, noise_level=0.1):
    transmission_db = np.zeros_like(frequency)
    center_freq1 = 0.870 + (r1 - 2.5) * 0.05 + (w - 2.5) * 0.03
    min_transmission1 = -12.657 + (r2 - 2.5) * 1.5 - (g - 2.5) * 1.0
    width1 = 0.08 + abs((r1 - 2.5) * 0.02)
    dip1_values = min_transmission1 * np.exp(-((frequency - center_freq1)**2) / (2 * width1**2))
    transmission_db += dip1_values
    center_freq2 = 2.115 + (r2 - 2.5) * 0.07 + (g - 2.5) * 0.04
    min_transmission2 = -11.763 + (r1 - 2.5) * 1.0 - (w - 2.5) * 0.8
    width2 = 0.15 + abs((r2 - 2.5) * 0.03)
    dip2_values = min_transmission2 * np.exp(-((frequency - center_freq2)**2) / (2 * width2**2))
    transmission_db += dip2_values
    transmission_db += -0.5 * (np.tanh((frequency - 1.5) * 2) + 1)
    if apply_offset:
        offset = -0.5 + 0.5 * (frequency / 3.0)
        transmission_db += offset
    noise = np.random.normal(0, noise_level, len(frequency))
    transmission_db += noise
    transmission_db = np.minimum(transmission_db, 0)
    peak_indices, _ = find_peaks(-transmission_db, prominence=2, width=3)
    f1, Q1, FoM1, S1 = np.nan, np.nan, np.nan, np.nan
    f2, Q2, FoM2, S2 = np.nan, np.nan, np.nan, np.nan
    if len(peak_indices) >= 1:
        idx1_closest = peak_indices[np.argmin(np.abs(frequency[peak_indices] - center_freq1))]
        f1, Q1, FoM1 = calculate_peak_parameters(frequency, transmission_db, idx1_closest)
        S1 = (f1 / 1.0) * (Q1 / 100.0) * 100 if not np.isnan(Q1) else np.nan
    if len(peak_indices) >= 2:
        remaining_indices = [idx for idx in peak_indices if idx != idx1_closest]
        if remaining_indices:
            idx2_closest = remaining_indices[np.argmin(np.abs(frequency[remaining_indices] - center_freq2))]
            f2, Q2, FoM2 = calculate_peak_parameters(frequency, transmission_db, idx2_closest)
            S2 = (f2 / 1.0) * (Q2 / 100.0) * 100 if not np.isnan(Q2) else np.nan
    if np.isnan(f1): f1 = center_freq1
    if np.isnan(f2): f2 = center_freq2
    return transmission_db, f1, f2, Q1, FoM1, S1, Q2, FoM2, S2

# --- PyTorch Dataset 类 ---
class MetamaterialDataset(Dataset):
    def __init__(self, data_path: str, num_points_per_sample: int = 250, load_data: bool = True):
        """
        初始化数据集。
        Args:
            data_path (str): CSV 数据文件的完整路径。
            num_points_per_sample (int): 每个光谱数据点数量。
            load_data (bool): 是否加载实际数据。如果为 False，则只加载元数据（如参数名、指标名）。
        """
        self.frequencies = np.linspace(0.5, 3.0, num_points_per_sample)
        self.param_ranges = {
            'r1': (2.2, 2.8), 'r2': (2.2, 2.8), 'w': (2.2, 2.8), 'g': (2.2, 2.8)
        }
        self.metric_names = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']

        self.spectrum_cols = [f'Freq_{f:.2f}' for f in self.frequencies]
        self.param_cols = ['r1', 'r2', 'w', 'g']
        self.metric_cols = self.metric_names
        
        self.spectra = None
        self.parameters = None
        self.metrics = None
        self.normalized_parameters = None
        self.normalized_metrics = None
        self.metric_min_max = {}
        self.metric_name_to_idx = {name: i for i, name in enumerate(self.metric_names)}

        if load_data:
            # --- 只从 CSV 加载数据 ---
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at: {data_path}. Please ensure the CSV file exists.")

            print(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)
            
            # 检查必要的列是否存在
            required_cols = self.spectrum_cols + self.param_cols + self.metric_cols
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in CSV: {missing_cols}. Please check your data file.")

            self.spectra = torch.tensor(df[self.spectrum_cols].values, dtype=torch.float32)
            self.parameters = torch.tensor(df[self.param_cols].values, dtype=torch.float32)
            self.metrics = torch.tensor(df[self.metric_cols].values, dtype=torch.float32)
            print(f"Loaded {len(self.spectra)} samples.")
            
            # --- 归一化处理 ---
            self.normalized_parameters = self.parameters.clone()
            for i, param_name in enumerate(self.param_cols):
                min_val, max_val = self.param_ranges[param_name]
                self.normalized_parameters[:, i] = (self.parameters[:, i] - min_val) / (max_val - min_val)

            self.normalized_metrics = self.metrics.clone()
            
            # 动态计算指标的min_max用于归一化
            for i, metric_name in enumerate(self.metric_names):
                # 过滤掉NaN值，避免影响min/max计算
                valid_metrics = self.metrics[~torch.isnan(self.metrics[:, i]), i]
                if len(valid_metrics) > 0:
                    min_val = valid_metrics.min().item()
                    max_val = valid_metrics.max().item()
                else: # 如果所有值都是NaN，给一个默认范围避免错误
                    min_val, max_val = 0.0, 1.0 # 默认值，可能需要根据实际数据调整

                self.metric_min_max[metric_name] = (min_val, max_val)

                if max_val - min_val > 1e-6:
                    self.normalized_metrics[:, i] = (self.metrics[:, i] - min_val) / (max_val - min_val)
                else:
                    self.normalized_metrics[:, i] = 0.5 # 如果范围为0，设为中间值

            # 处理NaN值：将NaN设置为0.5 (归一化后的中间值) 或其他合适的值
            # 确保在归一化后，NaN值不会传播
            self.normalized_metrics[torch.isnan(self.normalized_metrics)] = 0.5


    def __len__(self):
        if self.spectra is not None:
            return len(self.spectra)
        return 0 # 如果没有加载数据，长度为0

    def __getitem__(self, idx):
        if self.spectra is None:
            raise RuntimeError("Dataset not loaded. Set 'load_data=True' during initialization.")
        return (self.spectra[idx],
                self.parameters[idx],          # 未归一化的结构参数 (给 Discriminator)
                self.normalized_parameters[idx], # 归一化的结构参数 (给 Generator 和 ForwardModel)
                self.metrics[idx],             # 未归一化的物理指标 (备用)
                self.normalized_metrics[idx])  # 归一化的物理指标 (给 ForwardModel 损失计算)

# 定义反归一化函数 (用于结构参数)
def denormalize_params(norm_params_tensor: torch.Tensor, param_ranges: dict) -> torch.Tensor:
    denorm_params = torch.zeros_like(norm_params_tensor)
    param_names = ['r1', 'r2', 'w', 'g'] # 确保这里的顺序和数量与数据集中的参数一致
    for i, param_name in enumerate(param_names):
        min_val, max_val = param_ranges[param_name]
        denorm_params[:, i] = norm_params_tensor[:, i] * (max_val - min_val) + min_val
    return denorm_params

# 定义反归一化函数 (用于物理指标)
def denormalize_metrics(norm_metrics_tensor: torch.Tensor, metric_min_max: dict) -> torch.Tensor:
    """
    反归一化物理指标。

    Args:
        norm_metrics_tensor (torch.Tensor): 归一化后的物理指标张量。
        metric_min_max (dict): 包含每个指标原始 min/max 值的字典。
                                键为指标名称，值为 (min_val, max_val) 元组。
    Returns:
        torch.Tensor: 反归一化后的物理指标张量。
    """
    denorm_metrics = torch.zeros_like(norm_metrics_tensor)
    # 这里的循环需要知道指标的顺序，如果传入的norm_metrics_tensor维度是 (batch_size, num_metrics)，
    # 并且 metric_min_max 是按名称索引的，我们需要一个名称列表来迭代。
    # 假设 metric_min_max 的键顺序与 MetamaterialDataset 中 metric_names 的顺序一致。
    
    # 更好的做法是，当调用此函数时，明确传入 metric_names 列表，以便按正确顺序迭代
    # 例如：denormalize_metrics(norm_metrics, dataset.metric_min_max, dataset.metric_names)
    
    # 临时解决方案：从 metric_min_max 字典中获取有序的键
    # 注意：Python 3.7+ 字典保持插入顺序
    metric_names_ordered = list(metric_min_max.keys()) 
    
    for i, metric_name in enumerate(metric_names_ordered):
        min_val, max_val = metric_min_max[metric_name]
        if max_val - min_val > 1e-6: # 避免除以零或极小值
            denorm_metrics[:, i] = norm_metrics_tensor[:, i] * (max_val - min_val) + min_val
        else:
            denorm_metrics[:, i] = min_val # 如果范围为零，则所有归一化值都应映射回该单一值

    # 处理反归一化后可能出现的 NaN 值（如果原始归一化值是0.5但对应范围为0）
    denorm_metrics[torch.isnan(denorm_metrics)] = 0.0 # 根据您的数据特性选择一个合适的默认值

    return denorm_metrics


# --- 新增的 normalize_spectrum 函数 ---
def normalize_spectrum(spectrum_tensor: torch.Tensor, global_min_val: float = None, global_max_val: float = None) -> torch.Tensor:
    """
    对光谱数据进行归一化到 [0, 1] 范围。
    
    Args:
        spectrum_tensor (torch.Tensor): 原始光谱数据张量。
        global_min_val (float, optional): 数据集的全局最小值。如果提供，将使用此值进行归一化。
                                        如果为None，则使用当前张量的最小值。
        global_max_val (float, optional): 数据集的全局最大值。如果提供，将使用此值进行归一化。
                                        如果为None，则使用当前张量的最大值。

    Returns:
        torch.Tensor: 归一化后的光谱数据张量。
    """
    if global_min_val is not None and global_max_val is not None:
        min_val = global_min_val
        max_val = global_max_val
    else:
        # 如果未提供全局 min/max，则使用当前张量中的 min/max。
        # 建议在数据加载时，从整个训练集计算全局 min/max 以保持一致性。
        min_val = spectrum_tensor.min().item()
        max_val = spectrum_tensor.max().item()

    if max_val - min_val > 1e-8: # 避免除以零或极小值
        normalized_spectrum = (spectrum_tensor - min_val) / (max_val - min_val)
    else:
        # 如果所有值都相同（范围为零），则归一化为0.5
        normalized_spectrum = torch.full_like(spectrum_tensor, 0.5) 
    
    # 将归一化后的光谱裁剪到 [0, 1] 范围，以防浮点误差导致超出范围
    normalized_spectrum = torch.clamp(normalized_spectrum, 0.0, 1.0)
    
    return normalized_spectrum
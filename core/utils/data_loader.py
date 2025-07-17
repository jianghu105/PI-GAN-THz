# PI_GAN_THZ/core/utils/data_loader.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import find_peaks
import os
import config.config as cfg # 假设你的config文件名为config.py

# --- 辅助函数：计算Q参数、FoM参数、灵敏度S ---
# （这部分与你提供的代码相同，不做修改）
def calculate_peak_parameters(frequency, transmission_db, peak_idx, baseline_transmission=0):
    f_res = frequency[peak_idx]
    t_min = transmission_db[peak_idx]
    half_depth_val = t_min + (baseline_transmission - t_min) / 2
    f_lower, f_upper = np.nan, np.nan # 初始化为np.nan
    
    # 查找f_lower
    for i in range(peak_idx - 1, -1, -1):
        if (transmission_db[i] >= half_depth_val and transmission_db[i+1] < half_depth_val) or \
           (transmission_db[i] < half_depth_val and transmission_db[i+1] >= half_depth_val):
            # 线性插值
            if (transmission_db[i+1] - transmission_db[i]) != 0: # 避免除以零
                f_lower = frequency[i] + (half_depth_val - transmission_db[i]) * \
                          (frequency[i+1] - frequency[i]) / (transmission_db[i+1] - transmission_db[i])
            else:
                f_lower = frequency[i] # 如果水平线，直接取点
            break
    
    # 查找f_upper
    for i in range(peak_idx + 1, len(frequency) - 1): # 确保 i+1 不越界
        if (transmission_db[i] <= half_depth_val and transmission_db[i+1] > half_depth_val) or \
           (transmission_db[i] > half_depth_val and transmission_db[i+1] <= half_depth_val):
            # 线性插值
            if (transmission_db[i+1] - transmission_db[i]) != 0: # 避免除以零
                f_upper = frequency[i] + (half_depth_val - transmission_db[i]) * \
                          (frequency[i+1] - frequency[i]) / (transmission_db[i+1] - transmission_db[i])
            else:
                f_upper = frequency[i] # 如果水平线，直接取点
            break
            
    Q = np.nan
    FoM = np.nan
    
    # 只有当f_lower和f_upper都找到且有效时才计算Q和FoM
    if f_lower is not None and f_upper is not None and not np.isnan(f_lower) and not np.isnan(f_upper) and f_upper > f_lower:
        delta_f = f_upper - f_lower
        if delta_f > 1e-9: # 避免除以零或极小值
            Q = f_res / delta_f
        
        # 修正FoM的计算，确保分母不为零且有意义
        if not np.isnan(t_min) and abs(t_min) > 1e-6: # 使用一个小的 epsilon
            FoM = Q / abs(t_min) if not np.isnan(Q) else np.nan
        else:
            FoM = np.nan
            
    return f_res, Q, FoM

# --- 数据生成函数 (不再使用，但保留以防万一或作为参考) ---
# （这部分与你提供的代码相同，不做修改）
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
    
    # 查找峰值
    # 注意: find_peaks 期望正向峰值，所以我们对 -transmission_db 查找
    peak_indices, _ = find_peaks(-transmission_db, prominence=1.0, width=1) # 降低prominence和width以确保能找到更多峰
    
    f1, Q1, FoM1, S1 = np.nan, np.nan, np.nan, np.nan
    f2, Q2, FoM2, S2 = np.nan, np.nan, np.nan, np.nan

    # 尝试找到最接近预期中心频率的峰
    if len(peak_indices) > 0:
        # 第一个峰
        idx1_closest_candidates = peak_indices[np.argsort(np.abs(frequency[peak_indices] - center_freq1))]
        if len(idx1_closest_candidates) > 0:
            idx1_closest = idx1_closest_candidates[0]
            f1, Q1, FoM1 = calculate_peak_parameters(frequency, transmission_db, idx1_closest)
            S1 = (f1 / 1.0) * (Q1 / 100.0) * 100 if not np.isnan(Q1) else np.nan # 假设1.0和100.0是参考值

        # 第二个峰 (排除第一个峰的索引)
        remaining_indices = [idx for idx in peak_indices if idx != idx1_closest]
        if len(remaining_indices) > 0:
            idx2_closest_candidates = np.array(remaining_indices)[np.argsort(np.abs(frequency[remaining_indices] - center_freq2))]
            if len(idx2_closest_candidates) > 0:
                idx2_closest = idx2_closest_candidates[0]
                f2, Q2, FoM2 = calculate_peak_parameters(frequency, transmission_db, idx2_closest)
                S2 = (f2 / 1.0) * (Q2 / 100.0) * 100 if not np.isnan(Q2) else np.nan # 假设1.0和100.0是参考值
                
    # 如果没有找到有效的f1或f2，使用默认值
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
        
        # 硬编码的参数范围 (你的原始代码如此，可以根据需要从config加载)
        self.param_ranges = {
            'r1': (2.2, 2.8), 'r2': (2.2, 2.8), 'w': (2.2, 2.8), 'g': (2.2, 2.8)
        }
        
        self.metric_names = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
        # 注意：这里我们**不**硬编码 metric_ranges，因为它将从数据中动态计算
        # self.metric_ranges = {} # 将在加载数据时填充
        
        self.spectrum_cols = [f'Freq_{f:.2f}' for f in self.frequencies]
        self.param_cols = ['r1', 'r2', 'w', 'g']
        self.metric_cols = self.metric_names # 与 metric_names 保持一致
        
        self.spectra = None
        self.parameters = None
        self.metrics = None
        self.normalized_parameters = None
        self.normalized_metrics = None
        
        # 将 metric_min_max 重命名为 metric_ranges，以便与外部调用保持一致
        self.metric_ranges = {} # <--- **关键修改：重命名并初始化为字典**
        self.metric_name_to_idx = {name: i for i, name in enumerate(self.metric_names)}

        if load_data:
            # --- 只从 CSV 加载数据 ---
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"数据文件未找到: {data_path}. 请确保 CSV 文件存在。")

            print(f"正在从 {data_path} 加载数据...")
            df = pd.read_csv(data_path)
            
            # 检查必要的列是否存在
            # 确保光谱列在df中存在
            actual_spectrum_cols = [col for col in df.columns if col.startswith('Freq_') and len(col.split('_')) == 2 and col.split('_')[1].replace('.', '', 1).isdigit()]
            if not actual_spectrum_cols:
                raise ValueError("CSV中未找到任何以 'Freq_' 开头的光谱列。请检查数据格式。")
            
            # 使用实际找到的光谱列名来设置 self.spectrum_cols，并进行排序
            self.spectrum_cols = sorted(actual_spectrum_cols, key=lambda x: float(x.split('_')[1]))
            
            if len(self.spectrum_cols) != num_points_per_sample:
                 print(f"警告: CSV中找到的光谱点数 ({len(self.spectrum_cols)}) 与配置中的期望点数 ({num_points_per_sample}) 不匹配。将使用 CSV 中的实际点数。")
                 self.frequencies = np.linspace(0.5, 3.0, len(self.spectrum_cols)) # 根据实际点数调整频率

            required_param_cols = [col for col in self.param_cols if col not in df.columns]
            required_metric_cols = [col for col in self.metric_cols if col not in df.columns]

            if required_param_cols:
                raise ValueError(f"CSV中缺少所需的参数列: {required_param_cols}. 请检查数据文件。")
            if required_metric_cols:
                raise ValueError(f"CSV中缺少所需的指标列: {required_metric_cols}. 请检查数据文件。")

            self.spectra = torch.tensor(df[self.spectrum_cols].values, dtype=torch.float32)
            self.parameters = torch.tensor(df[self.param_cols].values, dtype=torch.float32)
            self.metrics = torch.tensor(df[self.metric_cols].values, dtype=torch.float32)
            print(f"已加载 {len(self.spectra)} 个样本。")
            
            # --- 归一化处理 ---
            # 参数归一化
            self.normalized_parameters = self.parameters.clone()
            for i, param_name in enumerate(self.param_cols):
                min_val, max_val = self.param_ranges[param_name] # 从硬编码的param_ranges中获取
                # 归一化到 [0, 1]
                if max_val - min_val > 1e-6:
                    self.normalized_parameters[:, i] = (self.parameters[:, i] - min_val) / (max_val - min_val)
                else:
                    self.normalized_parameters[:, i] = 0.5 # 如果范围为0，设为中间值
            # 进一步归一化到 [-1, 1] 适合GANs
            self.normalized_parameters = self.normalized_parameters * 2.0 - 1.0


            # 指标归一化
            self.normalized_metrics = self.metrics.clone()
            
            # 动态计算指标的min_max并存储到 self.metric_ranges
            for i, metric_name in enumerate(self.metric_names): # 确保使用 metric_names 来遍历
                # 过滤掉NaN值，避免影响min/max计算
                valid_metrics = self.metrics[~torch.isnan(self.metrics[:, i]), i]
                if len(valid_metrics) > 0:
                    min_val = valid_metrics.min().item()
                    max_val = valid_metrics.max().item()
                else: # 如果所有值都是NaN，给一个默认范围避免错误
                    min_val, max_val = 0.0, 1.0 # 默认值，可能需要根据实际数据调整
                
                # 存储到 metric_ranges 字典
                self.metric_ranges[metric_name] = (min_val, max_val) # <--- **关键修改：存储到 metric_ranges**

                if max_val - min_val > 1e-6: # 避免除以零
                    self.normalized_metrics[:, i] = (self.metrics[:, i] - min_val) / (max_val - min_val)
                else:
                    self.normalized_metrics[:, i] = 0.5 # 如果范围为0，设为中间值

            # 处理NaN值：将NaN设置为0.5 (归一化后的中间值) 或其他合适的值
            self.normalized_metrics[torch.isnan(self.normalized_metrics)] = 0.5


    def __len__(self):
        if self.spectra is not None:
            return len(self.spectra)
        return 0 # 如果没有加载数据，长度为0

    def __getitem__(self, idx):
        if self.spectra is None:
            raise RuntimeError("数据集未加载。请在初始化时设置 'load_data=True'。")
        return (self.spectra[idx],
                self.parameters[idx],          # 未归一化的结构参数 (给 Discriminator/绘图用)
                self.normalized_parameters[idx], # 归一化的结构参数 (给 Generator 和 ForwardModel)
                self.metrics[idx],             # 未归一化的物理指标 (绘图用)
                self.normalized_metrics[idx])  # 归一化的物理指标 (给 ForwardModel 损失计算)

# 定义反归一化函数 (用于结构参数)
# 注意：这些反归一化函数在 MetamaterialDataset 外部，需要通过 dataset.param_ranges 或 dataset.metric_ranges 访问
def denormalize_params(norm_params_tensor: torch.Tensor, param_ranges: dict) -> torch.Tensor:
    denorm_params = torch.zeros_like(norm_params_tensor)
    # 这里的param_names列表必须与你param_ranges字典的键顺序一致，
    # 或者从 param_ranges.keys() 获取并排序。
    # 鉴于你的param_ranges是硬编码的，我们可以直接使用硬编码的顺序
    param_names_ordered = ['r1', 'r2', 'w', 'g'] 

    for i, param_name in enumerate(param_names_ordered):
        min_val, max_val = param_ranges[param_name] # 从传入的param_ranges字典中获取
        # 反归一化：从 [0, 1] 映射回原始范围
        # 注意：如果你的normalized_parameters是 [-1, 1]，这里需要调整
        # (x + 1) / 2 将 [-1, 1] 映射到 [0, 1]
        norm_val_0_1 = (norm_params_tensor[:, i] + 1.0) / 2.0 
        denorm_params[:, i] = norm_val_0_1 * (max_val - min_val) + min_val
    return denorm_params

# 定义反归一化函数 (用于物理指标)
def denormalize_metrics(norm_metrics_tensor: torch.Tensor, metric_ranges: dict) -> torch.Tensor:
    """
    反归一化物理指标。

    Args:
        norm_metrics_tensor (torch.Tensor): 归一化后的物理指标张量。
        metric_ranges (dict): 包含每个指标原始 min/max 值的字典。
                              键为指标名称，值为 (min_val, max_val) 元组。
    Returns:
        torch.Tensor: 反归一化后的物理指标张量。
    """
    denorm_metrics = torch.zeros_like(norm_metrics_tensor)
    
    # 获取有序的指标名称列表，与 MetamaterialDataset 的 metric_names 顺序一致
    # 注意：如果 metric_ranges 是从 MetamaterialDataset.metric_names 动态填充的，
    # 那么 list(metric_ranges.keys()) 可能不保持原始顺序。
    # 最好是在 MetamaterialDataset 外部定义一个全局或通过参数传入的 metric_names 列表。
    # 考虑到 MetamaterialDataset 内部已经有了 self.metric_names，我们假设它是正确的顺序。
    # 所以，这里应该根据 MetamaterialDataset.metric_names 的顺序来处理。
    # 为了通用性，假定 `metric_ranges` 传入的键已经能按正确的顺序访问，
    # 或显式传入 `metric_names_order` 参数。
    # 对于当前场景，我们知道 metric_ranges 是通过 metric_names 填充的。
    metric_names_ordered = list(metric_ranges.keys()) # 如果Python版本 >= 3.7，字典会保持插入顺序

    for i, metric_name in enumerate(metric_names_ordered):
        min_val, max_val = metric_ranges[metric_name]
        if max_val - min_val > 1e-6: # 避免除以零或极小值
            # 归一化是 [0, 1] 区间，所以直接反归一化
            denorm_metrics[:, i] = norm_metrics_tensor[:, i] * (max_val - min_val) + min_val
        else:
            denorm_metrics[:, i] = min_val # 如果范围为零，则所有归一化值都应映射回该单一值

    # 处理反归一化后可能出现的 NaN 值
    # 如果原始 normalized_metrics 中有 0.5 (表示 NaN)，反归一化后会变成 min_val + 0.5 * (max_val - min_val)
    # 如果希望 NaN 仍然是 NaN，或者映射为其他特定值，需要更复杂的逻辑。
    # 目前保持 NaN 传递，或将其映射为 0.0
    denorm_metrics[torch.isnan(denorm_metrics)] = 0.0 # 根据您的数据特性选择一个合适的默认值，或保持 NaN

    return denorm_metrics


# --- 新增的 normalize_spectrum 函数 ---
# （这部分与你提供的代码相同，不做修改）
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
# PI_GAN_THZ/core/utils/physics_constraints.py

import torch
import torch.nn.functional as F
from scipy.signal import find_peaks

# --- 1. 核心物理模型 ---

def calculate_srr_frequencies(r1, r2, w, g):
    """
    计算十字辐条SRR的理论共振频率
    
    物理原理:
      - 基于耦合LC电路模型
      - 考虑十字辐条增强耦合(k)
      - 45°开口的边缘场效应
      - 金层0.2μm固定(趋肤深度已知)
    
    参数:
      r1, r2, w, g: 结构参数(单位: 米)
    
    返回:
      f1_theory, f2_theory: 理论共振频率(单位: Hz)
    """
    # 1. 计算等效电感(考虑十字辐条增强)
    mu0 = 4e-7 * torch.pi
    r_avg = (r1 + r2) / 2
    L_eff = mu0 * r_avg * (torch.log(8 * r_avg / w) - 1.75)
    
    # 2. 计算等效电容(考虑4个开口和45°角度)
    epsilon0 = 8.85e-12
    C_eff = 4 * epsilon0 * (r1 - r2) * g / (g + 0.7 * (r1 - r2))
    
    # 3. 计算耦合系数(十字辐条增强)
    k = 0.4 * w / (r1 - r2)  # 耦合强度
    
    # 4. 计算理论共振频率
    f0 = 1 / (2 * torch.pi * torch.sqrt(L_eff * C_eff))
    f1_theory = (1 - 0.45 * k) * f0
    f2_theory = (1 + 0.55 * k) * f0
    
    # 5. 45°开口经验修正
    f1_theory = f1_theory * 0.95
    f2_theory = f2_theory * 1.02
    
    return f1_theory, f2_theory

# --- 2. 生成器物理损失函数 ---

def lc_frequency_constraint(params, f1_pred, f2_pred):
    """
    LC频率约束损失 - 确保生成参数满足物理共振关系
    
    物理原理:
      - 基于您提供的数据: 主峰0.85THz, 次峰2.05THz
      - f2/f1 ≈ 2.41 (2.05/0.85)
      - 金层0.2μm固定简化了Q因子模型
    
    参数:
      params: [batch, 4] 生成的结构参数 [r1, r2, w, g]
      f1_pred, f2_pred: [batch] 预测的共振频率(归一化)
    
    返回:
      loss: 标量, LC频率约束损失
    """
    r1, r2, w, g = params.unbind(1)
    
    # 1. 计算理论频率
    f1_theory, f2_theory = calculate_srr_frequencies(r1, r2, w, g)
    
    # 2. 归一化到您的频率范围(0.5-3THz)
    f_min, f_max = 0.5e12, 3.0e12
    f1_theory_norm = (f1_theory - f_min) / (f_max - f_min)
    f2_theory_norm = (f2_theory - f_min) / (f_max - f_min)
    
    # 3. 相对误差(对物理量更合理)
    f1_error = torch.abs(f1_pred - f1_theory_norm) / (f1_theory_norm + 1e-12)
    f2_error = torch.abs(f2_pred - f2_theory_norm) / (f2_theory_norm + 1e-12)
    
    return (f1_error + f2_error).mean()

def maxwell_constraint(params, s1_pred, s2_pred):
    """
    麦克斯韦约束损失 - 确保双峰关系符合物理规律
    
    物理原理:
      - 基于您提供的数据: 主峰-25dB, 次峰-13dB
      - f2/f1 = 2.05/0.85 = 2.41
      - S1/S2 = 10^((-25+13)/20) = 0.063 (-12dB)
      - 金层0.2μm固定简化了损耗模型
    
    参数:
      params: [batch, 4] 生成的结构参数 [r1, r2, w, g]
      s1_pred, s2_pred: [batch] 预测的透射强度(线性值)
    
    返回:
      loss: 标量, 麦克斯韦约束损失
    """
    r1, r2, w, g = params.unbind(1)
    
    # 1. 计算理论双峰关系 (基于您的数据)
    expected_ratio = 2.05 / 0.85  # f2/f1 = 2.41
    
    # 2. 从参数计算实际f2/f1
    f1, f2 = calculate_srr_frequencies(r1, r2, w, g)
    actual_ratio = f2 / f1
    
    # 3. 双峰关系损失
    ratio_loss = torch.abs(actual_ratio - expected_ratio) / expected_ratio
    
    # 4. 能量比约束 (基于您的-25dB/-13dB)
    expected_energy_ratio = 10**((13-25)/20)  # -12dB = 0.063
    actual_energy_ratio = s1_pred / s2_pred
    # 使用log防止数值不稳定
    energy_loss = torch.abs(
        torch.log10(actual_energy_ratio + 1e-5) - 
        torch.log10(torch.tensor(expected_energy_ratio, device=s1_pred.device))
    )
    
    return ratio_loss.mean() + 0.5 * energy_loss.mean()

def energy_conservation_constraint(q1_pred, q2_pred):
    """
    能量守恒约束损失 - 确保Q因子比值合理
    
    物理原理:
      - 基于金层0.2μm固定的欧姆损耗模型
      - 主峰(0.85THz)损耗大 → Q1小
      - 次峰(2.05THz)损耗小 → Q2大
      - Q1/Q2 ≈ 0.55 (来自您的-25dB/-13dB)
    
    参数:
      q1_pred, q2_pred: [batch] 预测的Q因子
    
    返回:
      loss: 标量, 能量守恒约束损失
    """
    # 1. 基于您的数据的期望Q比值
    expected_q_ratio = 0.55  # Q1/Q2
    
    # 2. 计算实际Q比值
    actual_q_ratio = q1_pred / (q2_pred + 1e-5)
    
    # 3. Q比值损失
    q_ratio_loss = torch.abs(actual_q_ratio - expected_q_ratio) / expected_q_ratio
    
    # 4. Q因子范围约束(您的数据范围)
    q1_range_loss = torch.relu(10.0 - q1_pred) + torch.relu(q1_pred - 50.0)
    q2_range_loss = torch.relu(5.0 - q2_pred) + torch.relu(q2_pred - 30.0)
    
    return q_ratio_loss.mean() + 0.2 * (q1_range_loss.mean() + q2_range_loss.mean())

# --- 3. 验证与监控辅助函数 ---

def find_peak_frequencies(spectrum, f_min, f_max):
    """从光谱中提取共振峰频率"""
    # spectrum: [batch, 250], 频率范围已知
    freqs = torch.linspace(0.1e12, 3.0e12, 250, device=spectrum.device)
    
    peaks_list = []
    for i in range(spectrum.shape[0]):
        # 找透射率极小值点(共振峰)
        spec = spectrum[i].detach().cpu().numpy()
        # 使用scipy.signal.find_peaks
        min_peaks_indices, _ = find_peaks(-spec, distance=20)
        
        # 筛选在目标频段内的峰
        valid_peaks = [freqs[p] for p in min_peaks_indices 
                      if f_min <= freqs[p] <= f_max]
        
        if valid_peaks:
            peaks_list.append(valid_peaks[0])  # 取第一个主峰
        else:
            # 如果找不到峰，使用频段中心作为默认值
            peaks_list.append(torch.tensor((f_min + f_max) / 2, device=spectrum.device))
    
    return torch.stack(peaks_list)

def extract_q_factor(spectrum, f_res, min_q=5, max_q=50):
    """从光谱中提取Q因子"""
    freqs = torch.linspace(0.1e12, 3.0e12, 250, device=spectrum.device)
    
    q_factors = []
    for i in range(spectrum.shape[0]):
        spec_i = spectrum[i]
        f_res_i = f_res[i]
        
        # 找到共振频率对应的索引
        res_idx = torch.argmin(torch.abs(freqs - f_res_i))
        res_val = spec_i[res_idx]
        
        # 计算-3dB点的值 (线性尺度)
        # S_3dB = S_res + 0.5 * (1 - S_res)
        # 在对数尺度上是 S_res_dB + 3dB
        # 这里用线性尺度
        half_max_val = res_val + 0.5 * (1 - res_val)

        # 向左找-3dB点
        left_idx = res_idx
        while left_idx > 0 and spec_i[left_idx] < half_max_val:
            left_idx -= 1
        
        # 向右找-3dB点
        right_idx = res_idx
        while right_idx < (len(freqs) - 1) and spec_i[right_idx] < half_max_val:
            right_idx += 1
        
        # 计算带宽
        bandwidth = freqs[right_idx] - freqs[left_idx]
        q = f_res_i / (bandwidth + 1e-12)
        
        # 限制在合理范围
        q_clamped = torch.clamp(q, min_q, max_q)
        q_factors.append(q_clamped)
    
    return torch.stack(q_factors)

def extract_transmission(spectrum, f_res):
    """从光谱中提取指定频率的透射强度"""
    freqs = torch.linspace(0.1e12, 3.0e12, 250, device=spectrum.device)
    transmissions = []
    for i in range(spectrum.shape[0]):
        res_idx = torch.argmin(torch.abs(freqs - f_res[i]))
        transmissions.append(spectrum[i, res_idx])
    return torch.stack(transmissions)

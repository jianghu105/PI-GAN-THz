import numpy as np
import pandas as pd
import re
import os
import time
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

# 文件路径
data_file = 'TMS233.txt'
output_file = 'thz_results2.csv'

# 常量定义
# 峰值检测参数
PEAK_PROMINENCE_PRIMARY = 2.0    # 主要峰值的突出度阈值
PEAK_PROMINENCE_SECONDARY = 1.0  # 次要峰值的突出度阈值
PEAK_HEIGHT_PRIMARY = 2.0        # 主要峰值的高度阈值
PEAK_HEIGHT_SECONDARY = 1.0      # 次要峰值的高度阈值
PEAK_DISTANCE = 50               # 峰值之间的最小距离

# Q因子计算参数
HALF_POWER_DB = 3.0              # 半功率点对应的dB值（3dB）

# 数据采样参数
SAMPLE_POINTS = 250              # 每组数据采样点数

def extract_parameters(param_line):
    """从参数行提取参数值"""
    pattern = r'\{(.+)\}'
    match = re.search(pattern, param_line)
    if match:
        param_str = match.group(1)
        params = {}
        for item in param_str.split(';'):
            key, value = item.strip().split('=')
            params[key.strip()] = float(value.strip())
        return params
    return None

def find_resonance_peaks(freq, mag):
    """找出共振峰的位置
    
    参数:
        freq (numpy.ndarray): 频率数组
        mag (numpy.ndarray): 幅度数组（dB单位）
        
    返回:
        tuple: (f1, f2, q1, q2, fom1, fom2, s1, s2) 包含两个共振峰的频率、Q因子、FoM和灵敏度
    """
    try:
        # 由于共振峰是负向峰（幅度最小值），我们取负值来寻找峰值
        neg_mag = -np.array(mag)
        
        # 数据预处理：平滑处理以减少噪声影响
        window_size = min(11, len(neg_mag) // 10)  # 窗口大小不超过数据长度的1/10
        if window_size % 2 == 0:  # 确保窗口大小为奇数
            window_size += 1
        if window_size >= 3:  # 只有当窗口大小足够大时才进行平滑
            try:
                from scipy.signal import savgol_filter
                neg_mag_smooth = savgol_filter(neg_mag, window_size, 2)  # 使用Savitzky-Golay滤波器
            except:
                # 如果savgol_filter不可用，使用简单的移动平均
                kernel = np.ones(window_size) / window_size
                neg_mag_smooth = np.convolve(neg_mag, kernel, mode='same')
        else:
            neg_mag_smooth = neg_mag
        
        # 自适应参数：根据数据范围调整prominence和height
        data_range = np.max(neg_mag_smooth) - np.min(neg_mag_smooth)
        adaptive_prominence = max(PEAK_PROMINENCE_PRIMARY, data_range * 0.1)  # 至少为数据范围的10%
        adaptive_height = max(PEAK_HEIGHT_PRIMARY, np.mean(neg_mag_smooth) + data_range * 0.2)  # 至少为平均值加数据范围的20%
        
        # 寻找主要的两个峰，使用自适应参数
        peaks, properties = find_peaks(neg_mag_smooth, 
                                      prominence=adaptive_prominence, 
                                      height=adaptive_height, 
                                      distance=PEAK_DISTANCE)
        
        if len(peaks) < 2:
            # 如果找不到足够的峰，尝试降低阈值
            adaptive_prominence = max(PEAK_PROMINENCE_SECONDARY, data_range * 0.05)  # 降低到数据范围的5%
            adaptive_height = max(PEAK_HEIGHT_SECONDARY, np.mean(neg_mag_smooth) + data_range * 0.1)  # 降低到平均值加数据范围的10%
            
            peaks, properties = find_peaks(neg_mag_smooth, 
                                          prominence=adaptive_prominence, 
                                          height=adaptive_height, 
                                          distance=PEAK_DISTANCE//2)
        
        if len(peaks) < 2:
            # 如果仍然找不到足够的峰，尝试只使用distance参数
            peaks, properties = find_peaks(neg_mag_smooth, distance=PEAK_DISTANCE//3)
        
        if len(peaks) < 2:
            # 如果仍然找不到足够的峰，返回空值
            return None, None, None, None, None, None, None, None
        
        # 按峰值大小排序
        peak_heights = neg_mag_smooth[peaks]
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_peaks = peaks[sorted_indices[:min(2, len(sorted_indices))]]
        
        # 确保峰按频率排序
        top_peaks = sorted(top_peaks)
        
        if len(top_peaks) < 2:
            return None, None, None, None, None, None, None, None
        
        f1_idx, f2_idx = top_peaks[0], top_peaks[1]
        f1, f2 = freq[f1_idx], freq[f2_idx]
        
        # 计算Q因子 (f_resonance / FWHM)
        q1 = calculate_q_factor(freq, mag, f1_idx)
        q2 = calculate_q_factor(freq, mag, f2_idx)
        
        # 计算FoM (Figure of Merit)
        fom1 = calculate_fom(q1, mag[f1_idx])
        fom2 = calculate_fom(q2, mag[f2_idx])
        
        # 计算灵敏度S
        s1 = calculate_sensitivity(mag[f1_idx])
        s2 = calculate_sensitivity(mag[f2_idx])
        
        return f1, f2, q1, q2, fom1, fom2, s1, s2
    except Exception as e:
        print(f"峰值检测出错: {e}")
        return None, None, None, None, None, None, None, None

def calculate_q_factor(freq, mag, peak_idx):
    """计算Q因子 (f_resonance / FWHM)
    
    参数:
        freq (numpy.ndarray): 频率数组
        mag (numpy.ndarray): 幅度数组（dB单位）
        peak_idx (int): 峰值在数组中的索引
        
    返回:
        float or None: Q因子值，如果无法计算则返回None
    """
    try:
        peak_freq = freq[peak_idx]
        peak_mag = mag[peak_idx]
        
        # 找到半高宽（FWHM）的点
        half_power = peak_mag + HALF_POWER_DB  # 在dB尺度上，半功率点是峰值+3dB
        
        # 在峰值左侧寻找半高宽点
        left_idx = peak_idx
        while left_idx > 0 and mag[left_idx] < half_power:
            left_idx -= 1
            
        # 在峰值右侧寻找半高宽点
        right_idx = peak_idx
        while right_idx < len(freq) - 1 and mag[right_idx] < half_power:
            right_idx += 1
        
        # 如果找不到半高宽点，尝试外推
        if left_idx == 0 or right_idx == len(freq) - 1:
            # 尝试使用线性外推
            if left_idx == 0 and len(freq) > 2:
                # 左侧外推
                slope = (mag[2] - mag[0]) / (freq[2] - freq[0])
                if slope != 0:
                    f_left = freq[0] - (half_power - mag[0]) / slope
                    if f_left <= 0 or not np.isfinite(f_left):
                        return None
                else:
                    return None
            else:
                # 通过插值找到更精确的左侧半高宽点
                f_left = interpolate_frequency(freq[left_idx], freq[left_idx+1], 
                                             mag[left_idx], mag[left_idx+1], half_power)
            
            if right_idx == len(freq) - 1 and len(freq) > 2:
                # 右侧外推
                slope = (mag[len(freq)-1] - mag[len(freq)-3]) / (freq[len(freq)-1] - freq[len(freq)-3])
                if slope != 0:
                    f_right = freq[len(freq)-1] + (half_power - mag[len(freq)-1]) / slope
                    if not np.isfinite(f_right):
                        return None
                else:
                    return None
            else:
                # 通过插值找到更精确的右侧半高宽点
                f_right = interpolate_frequency(freq[right_idx-1], freq[right_idx], 
                                              mag[right_idx-1], mag[right_idx], half_power)
        else:
            # 通过插值找到更精确的半高宽点
            f_left = interpolate_frequency(freq[left_idx], freq[left_idx+1], 
                                         mag[left_idx], mag[left_idx+1], half_power)
            f_right = interpolate_frequency(freq[right_idx-1], freq[right_idx], 
                                          mag[right_idx-1], mag[right_idx], half_power)
        
        fwhm = f_right - f_left
        if fwhm <= 0 or not np.isfinite(fwhm):
            return None
        
        q_factor = peak_freq / fwhm
        
        # 检查Q因子是否在合理范围内
        if not np.isfinite(q_factor) or q_factor <= 0 or q_factor > 1000:
            return None
            
        return q_factor
    except Exception as e:
        print(f"计算Q因子时出错: {e}")
        return None

def interpolate_frequency(f1, f2, m1, m2, target_m):
    """线性插值找到特定幅度对应的频率"""
    if m1 == m2:
        return f1
    return f1 + (f2 - f1) * (target_m - m1) / (m2 - m1)

def calculate_fom(q_factor, magnitude):
    """计算FoM (Figure of Merit)
    
    参数:
        q_factor (float): Q因子
        magnitude (float): 幅度值（dB单位）
        
    返回:
        float or None: FoM值，如果无法计算则返回None
    """
    try:
        if q_factor is None or not np.isfinite(q_factor) or q_factor <= 0:
            return None
            
        # 将dB转换为线性尺度的幅度
        linear_magnitude = 10 ** (abs(magnitude) / 20.0)
        
        # FoM定义为Q因子与线性尺度幅度的乘积
        fom = q_factor * linear_magnitude
        
        # 检查FoM是否在合理范围内
        if not np.isfinite(fom) or fom <= 0:
            return None
            
        return fom
    except Exception as e:
        print(f"计算FoM时出错: {e}")
        return None

def calculate_sensitivity(magnitude):
    """计算灵敏度S
    
    参数:
        magnitude (float): 幅度值（dB单位）
        
    返回:
        float or None: 灵敏度值，如果无法计算则返回None
    """
    try:
        if magnitude is None or not np.isfinite(magnitude):
            return None
            
        # 将dB转换为线性尺度
        linear_magnitude = 10 ** (abs(magnitude) / 20.0)
        
        # 灵敏度定义为线性尺度幅度的对数乘以常数因子
        # 这样可以更好地反映幅度变化的相对重要性
        sensitivity = np.log10(linear_magnitude) * 10
        
        # 确保灵敏度为正值
        sensitivity = max(0, sensitivity)
        
        # 检查灵敏度是否在合理范围内
        if not np.isfinite(sensitivity):
            return None
            
        return sensitivity
    except Exception as e:
        print(f"计算灵敏度时出错: {e}")
        return None

def process_data_file():
    """处理数据文件并提取所需信息"""
    print("开始处理数据文件...")
    start_time = time.time()
    
    try:
        # 使用with语句确保文件正确关闭
        with open(data_file, 'r') as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        print(f"读取了 {total_lines} 行数据")
        
        # 计算数据组数
        param_count = sum(1 for line in lines if line.strip().startswith('#Parameters'))
        print(f"数据文件包含 {param_count} 组数据")
        
        all_data = []
        current_params = None
        freq_data = []
        mag_data = []
        group_count = 0
        last_progress_report = time.time()
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # 每处理50000行报告一次进度
            if i % 50000 == 0 and i > 0:
                current_time = time.time()
                if current_time - last_progress_report >= 3:  # 至少间隔3秒报告一次
                    elapsed_time = current_time - start_time
                    progress_percent = i/total_lines*100
                    estimated_total_time = elapsed_time / (progress_percent/100) if progress_percent > 0 else 0
                    remaining_time = estimated_total_time - elapsed_time
                    
                    print(f"正在处理... 已完成 {i}/{total_lines} 行 ({progress_percent:.1f}%)")
                    print(f"已处理 {group_count}/{param_count} 组数据，预计剩余时间: {remaining_time/60:.1f} 分钟")
                    last_progress_report = current_time
            
            if line.startswith('#Parameters'):
                # 如果已经有数据，处理它
                if current_params and freq_data and mag_data:
                    process_group(current_params, freq_data, mag_data, all_data)
                    group_count += 1
                    # 每处理50组数据报告一次
                    if group_count % 50 == 0:
                        print(f"处理完成组 {group_count}/{param_count}")
                
                # 开始新的数据组
                current_params = extract_parameters(line)
                freq_data = []
                mag_data = []
            elif line.startswith('#'):
                # 跳过其他注释行
                continue
            else:
                # 解析数据行
                try:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        freq, mag = float(parts[0]), float(parts[1])
                        freq_data.append(freq)
                        mag_data.append(mag)
                except Exception as e:
                    # 减少错误输出，避免日志过多
                    if i % 10000 == 0:
                        print(f"解析数据行出错: 第 {i+1} 行, 错误: {e}")
                    continue
        
        # 处理最后一组数据
        if current_params and freq_data and mag_data:
            process_group(current_params, freq_data, mag_data, all_data)
            group_count += 1
            print(f"处理完成组 {group_count}/{param_count}")
        
        end_time = time.time()
        processing_time = end_time - start_time
        print(f"总共处理了 {group_count}/{param_count} 组数据，耗时 {processing_time:.2f} 秒 ({processing_time/60:.2f} 分钟)")
        
        # 创建DataFrame并保存结果
        if all_data:
            print(f"正在保存 {len(all_data)} 组数据到CSV文件...")
            save_start_time = time.time()
            
            # 创建基本列
            base_columns = ['r1', 'r2', 'w', 'g', 'f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
            
            # 创建频率列
            if len(all_data) > 0 and 'freq_data' in all_data[0]:
                freq_columns = [f'Freq_{freq:.2f}' for freq in all_data[0]['freq_data']]
                all_columns = base_columns + freq_columns
                
                # 创建新的数据列表，不包含freq_data和mag_data键
                clean_data = []
                for item in all_data:
                    clean_item = {k: v for k, v in item.items() if k != 'freq_data' and k != 'mag_data'}
                    # 添加频率数据
                    if 'mag_data' in item:
                        for i, freq in enumerate(item['freq_data']):
                            clean_item[f'Freq_{freq:.2f}'] = item['mag_data'][i]
                    clean_data.append(clean_item)
                
                df = pd.DataFrame(clean_data)
            else:
                df = pd.DataFrame(all_data)
            
            # 保存CSV文件
            df.to_csv(output_file, index=False, float_format='%.6f')
            save_end_time = time.time()
            save_time = save_end_time - save_start_time
            print(f"结果已保存到 {output_file}，保存耗时 {save_time:.2f} 秒")
            
            # 绘制第一组数据的图表作为示例
            if len(all_data) > 0:
                print("正在生成示例图表...")
                plot_example(all_data[0])
    
    except Exception as e:
        print(f"处理数据文件时出错: {e}")
        import traceback
        traceback.print_exc()
        # 如果已经处理了一些数据，尝试保存
        if 'all_data' in locals() and all_data:
            print(f"尝试保存已处理的 {len(all_data)} 组数据...")
            df = pd.DataFrame(all_data)
            df.to_csv(output_file, index=False, float_format='%.6f')
            print(f"已保存部分处理结果到 {output_file}")
        return

def process_group(params, freq_data, mag_data, all_data):
    """处理单个数据组并提取特征
    
    参数:
        params (dict): 参数字典，包含r1, r2, w, g等参数
        freq_data (list): 频率数据列表
        mag_data (list): 幅度数据列表
        all_data (list): 存储所有处理结果的列表
    """
    try:
        # 检查数据有效性
        if not freq_data or not mag_data or len(freq_data) != len(mag_data):
            print(f"警告: 数据无效或长度不匹配 (频率: {len(freq_data)}, 幅度: {len(mag_data)})")
            return
            
        # 转换为numpy数组以提高性能
        freq = np.array(freq_data, dtype=np.float64)
        mag = np.array(mag_data, dtype=np.float64)
        
        # 检查数据是否包含无效值
        if not np.all(np.isfinite(freq)) or not np.all(np.isfinite(mag)):
            # 过滤掉无效值
            valid_indices = np.logical_and(np.isfinite(freq), np.isfinite(mag))
            freq = freq[valid_indices]
            mag = mag[valid_indices]
            print(f"警告: 过滤了 {len(freq_data) - len(freq)} 个无效数据点")
            
            if len(freq) == 0:
                print("错误: 过滤后没有有效数据点")
                return
        
        # 找出共振峰并计算相关参数
        f1, f2, q1, q2, fom1, fom2, s1, s2 = find_resonance_peaks(freq, mag)
        
        # 创建结果字典
        result = {
            'r1': params.get('r1', None),
            'r2': params.get('r2', None),
            'w': params.get('w', None),
            'g': params.get('g', None),
            'f1': f1,
            'f2': f2,
            'Q1': q1,
            'FoM1': fom1,
            'S1': s1,
            'Q2': q2,
            'FoM2': fom2,
            'S2': s2
        }
        
        # 添加规律间隔的频率点数据
        # 从所有点中选择指定数量的点
        if len(freq) >= SAMPLE_POINTS * 2:  # 确保有足够的数据点
            # 使用更高效的采样方法
            step = max(1, len(freq) // SAMPLE_POINTS)
            
            if step > 1:
                # 如果步长大于1，使用步长采样（更高效）
                selected_freq = freq[::step]
                selected_mag = mag[::step]
                
                # 如果采样点过多，截断到SAMPLE_POINTS
                if len(selected_freq) > SAMPLE_POINTS:
                    selected_freq = selected_freq[:SAMPLE_POINTS]
                    selected_mag = selected_mag[:SAMPLE_POINTS]
                # 如果采样点不足，补充到SAMPLE_POINTS
                elif len(selected_freq) < SAMPLE_POINTS:
                    # 使用线性采样补充
                    indices = np.linspace(0, len(freq)-1, SAMPLE_POINTS, dtype=int)
                    selected_freq = freq[indices]
                    selected_mag = mag[indices]
            else:
                # 如果步长为1，直接使用线性采样
                indices = np.linspace(0, len(freq)-1, SAMPLE_POINTS, dtype=int)
                selected_freq = freq[indices]
                selected_mag = mag[indices]
        else:
            # 如果数据点不足，使用所有点
            selected_freq = freq
            selected_mag = mag
        
        # 存储选择的频率和幅度数据
        result['freq_data'] = selected_freq
        result['mag_data'] = selected_mag
        
        all_data.append(result)
    except Exception as e:
        print(f"处理数据组时出错: {e}")
        import traceback
        traceback.print_exc()

def plot_example(data_row, output_file='example_response.png'):
    """绘制示例频率响应曲线
    
    参数:
        data_row (dict): 包含频率、幅度和参数数据的字典
        output_file (str): 输出图像文件路径
    """
    try:
        # 提取频率和幅度数据
        if 'freq_data' in data_row and 'mag_data' in data_row:
            freq_values = data_row['freq_data']
            mag_values = data_row['mag_data']
        else:
            freq_cols = [col for col in data_row.keys() if col.startswith('Freq_')]
            freq_values = [float(col.split('_')[1]) for col in freq_cols]
            mag_values = [data_row[col] for col in freq_cols]
        
        if len(freq_values) == 0 or len(mag_values) == 0:
            print("频率或幅度数据为空，无法绘图")
            return
            
        # 检查数据有效性
        freq_values = np.array(freq_values)
        mag_values = np.array(mag_values)
        valid_indices = np.logical_and(np.isfinite(freq_values), np.isfinite(mag_values))
        freq_values = freq_values[valid_indices]
        mag_values = mag_values[valid_indices]
        
        if len(freq_values) == 0:
            print("过滤无效值后，频率或幅度数据为空，无法绘图")
            return
        
        # 设置matplotlib字体，避免中文显示问题
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        except Exception as e:
            print(f"设置字体时出错: {e}")
        
        # 创建高质量图表
        plt.figure(figsize=(14, 10), dpi=100)
        
        # 主曲线绘制，使用更美观的样式
        plt.plot(freq_values, mag_values, color='#1f77b4', linewidth=2.5, 
                 label='S参数', zorder=1, alpha=0.9)
        
        # 添加网格和背景
        plt.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
        plt.gca().set_facecolor('#f8f9fa')  # 设置浅灰色背景
        
        # 设置坐标轴标签和标题
        plt.xlabel('频率 (THz)', fontsize=14, fontweight='bold')
        plt.ylabel('幅度 (dB)', fontsize=14, fontweight='bold')
        
        # 构建标题，确保参数值格式化为字符串
        r1_str = f"{data_row['r1']}" if data_row['r1'] is not None else "N/A"
        r2_str = f"{data_row['r2']}" if data_row['r2'] is not None else "N/A"
        w_str = f"{data_row['w']}" if data_row['w'] is not None else "N/A"
        g_str = f"{data_row['g']}" if data_row['g'] is not None else "N/A"
        
        title = f"太赫兹超材料频率响应\nr1={r1_str}, r2={r2_str}, w={w_str}, g={g_str}"
        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        
        # 计算文本位置
        min_mag = np.min(mag_values)
        max_mag = np.max(mag_values)
        y_range = max_mag - min_mag
        text_y_pos = min_mag - y_range * 0.08  # 文本位置在最小值下方
        
        # 标记共振峰，使用更美观的样式
        if data_row['f1'] is not None:
            q1_str = f"{data_row['Q1']:.1f}" if data_row['Q1'] is not None else "N/A"
            # 绘制垂直线
            plt.axvline(x=data_row['f1'], color='#d62728', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2,
                        label=f"f1={data_row['f1']:.3f} THz, Q1={q1_str}")
            
            # 添加标记文本，使用文本框增强可读性
            plt.annotate(f"f1={data_row['f1']:.3f}", 
                         xy=(data_row['f1'], np.min(mag_values[freq_values > data_row['f1']*0.9])), 
                         xytext=(data_row['f1'], text_y_pos),
                         color='#d62728', fontweight='bold', ha='center', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#d62728", alpha=0.7),
                         zorder=3)
        
        if data_row['f2'] is not None:
            q2_str = f"{data_row['Q2']:.1f}" if data_row['Q2'] is not None else "N/A"
            # 绘制垂直线
            plt.axvline(x=data_row['f2'], color='#2ca02c', linestyle='--', 
                        linewidth=1.5, alpha=0.7, zorder=2,
                        label=f"f2={data_row['f2']:.3f} THz, Q2={q2_str}")
            
            # 添加标记文本，使用文本框增强可读性
            plt.annotate(f"f2={data_row['f2']:.3f}", 
                         xy=(data_row['f2'], np.min(mag_values[freq_values > data_row['f2']*0.9])), 
                         xytext=(data_row['f2'], text_y_pos),
                         color='#2ca02c', fontweight='bold', ha='center', va='top',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", alpha=0.7),
                         zorder=3)
        
        # 添加详细参数信息文本框
        info_text = []
        
        if data_row['f1'] is not None:
            fom1_str = f"{data_row['FoM1']:.2f}" if data_row['FoM1'] is not None else "N/A"
            s1_str = f"{data_row['S1']:.2f}" if data_row['S1'] is not None else "N/A"
            info_text.append(f"共振峰1: f1={data_row['f1']:.3f} THz, Q1={q1_str}, FoM1={fom1_str}, S1={s1_str}")
        
        if data_row['f2'] is not None:
            fom2_str = f"{data_row['FoM2']:.2f}" if data_row['FoM2'] is not None else "N/A"
            s2_str = f"{data_row['S2']:.2f}" if data_row['S2'] is not None else "N/A"
            info_text.append(f"共振峰2: f2={data_row['f2']:.3f} THz, Q2={q2_str}, FoM2={fom2_str}, S2={s2_str}")
        
        if info_text:
            # 创建一个更美观的文本框
            info_box = '\n'.join(info_text)
            plt.annotate(info_box, xy=(0.02, 0.02), xycoords='figure fraction',
                         fontsize=12, ha='left', va='bottom',
                         bbox=dict(boxstyle="round,pad=0.5", fc="#f8f8f8", 
                                   ec="#cccccc", alpha=0.9, linewidth=1.5),
                         zorder=4)
        
        # 添加图例
        plt.legend(loc='upper right', fontsize=12, framealpha=0.9, facecolor='#f8f8f8',
                   edgecolor='#cccccc', fancybox=True, shadow=True)
        
        # 优化布局
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # 保存高质量图像
        plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"示例图表已保存到 {output_file}")
        
    except Exception as e:
        print(f"绘制图表时出错: {e}")
        import traceback
        traceback.print_exc()
        # 尝试关闭图表以防止资源泄漏
        try:
            plt.close()
        except:
            pass

if __name__ == "__main__":
    process_data_file()
    print("处理完成！")
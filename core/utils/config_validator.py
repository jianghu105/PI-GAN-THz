"""
配置验证和自动边界检测模块
解决硬编码边界和配置不一致的问题
"""

import pandas as pd
import numpy as np
import torch
import os
import sys
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
from dataclasses import dataclass
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

logger = logging.getLogger(__name__)

@dataclass
class BoundaryInfo:
    """边界信息数据类"""
    min_value: float
    max_value: float
    range_value: float
    mean_value: float
    std_value: float
    percentile_5: float
    percentile_95: float
    recommended_min: float
    recommended_max: float

class AutoBoundaryDetector:
    """自动边界检测器"""
    
    def __init__(self, safety_margin: float = 0.05, percentile_range: Tuple[float, float] = (5, 95)):
        """
        Args:
            safety_margin: 安全边距比例
            percentile_range: 用于确定边界的百分位数范围
        """
        self.safety_margin = safety_margin
        self.percentile_range = percentile_range
        
    def detect_boundaries(self, data: pd.DataFrame, 
                         param_names: List[str]) -> Dict[str, BoundaryInfo]:
        """检测参数边界"""
        boundaries = {}
        
        for param in param_names:
            if param not in data.columns:
                warnings.warn(f"参数 {param} 不存在于数据中")
                continue
                
            param_data = data[param].dropna()
            
            if len(param_data) == 0:
                warnings.warn(f"参数 {param} 无有效数据")
                continue
            
            # 基本统计
            min_val = param_data.min()
            max_val = param_data.max()
            range_val = max_val - min_val
            mean_val = param_data.mean()
            std_val = param_data.std()
            
            # 百分位数
            p5 = np.percentile(param_data, self.percentile_range[0])
            p95 = np.percentile(param_data, self.percentile_range[1])
            
            # 推荐边界（考虑安全边距）
            margin = range_val * self.safety_margin
            rec_min = max(min_val - margin, p5 - margin)
            rec_max = min(max_val + margin, p95 + margin)
            
            boundaries[param] = BoundaryInfo(
                min_value=min_val,
                max_value=max_val,
                range_value=range_val,
                mean_value=mean_val,
                std_value=std_val,
                percentile_5=p5,
                percentile_95=p95,
                recommended_min=rec_min,
                recommended_max=rec_max
            )
            
            logger.info(f"参数 {param}: 范围=[{min_val:.4f}, {max_val:.4f}], "
                       f"推荐边界=[{rec_min:.4f}, {rec_max:.4f}]")
        
        return boundaries
    
    def validate_existing_boundaries(self, boundaries: Dict[str, BoundaryInfo],
                                   current_bounds: Dict[str, Tuple[float, float]]) -> Dict[str, Dict]:
        """验证现有边界设置"""
        validation_results = {}
        
        for param, boundary_info in boundaries.items():
            if param in current_bounds:
                current_min, current_max = current_bounds[param]
                
                # 检查是否过于严格
                too_strict_min = current_min > boundary_info.recommended_min
                too_strict_max = current_max < boundary_info.recommended_max
                
                # 检查是否过于宽松
                too_loose_min = current_min < boundary_info.min_value - boundary_info.range_value * 0.1
                too_loose_max = current_max > boundary_info.max_value + boundary_info.range_value * 0.1
                
                validation_results[param] = {
                    'current_bounds': (current_min, current_max),
                    'recommended_bounds': (boundary_info.recommended_min, boundary_info.recommended_max),
                    'data_bounds': (boundary_info.min_value, boundary_info.max_value),
                    'too_strict_min': too_strict_min,
                    'too_strict_max': too_strict_max,
                    'too_loose_min': too_loose_min,
                    'too_loose_max': too_loose_max,
                    'needs_adjustment': too_strict_min or too_strict_max or too_loose_min or too_loose_max
                }
                
                if validation_results[param]['needs_adjustment']:
                    logger.warning(f"参数 {param} 边界需要调整")
            else:
                validation_results[param] = {
                    'current_bounds': None,
                    'recommended_bounds': (boundary_info.recommended_min, boundary_info.recommended_max),
                    'data_bounds': (boundary_info.min_value, boundary_info.max_value),
                    'missing': True
                }
                logger.warning(f"参数 {param} 缺少边界设置")
        
        return validation_results

class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.validation_errors = []
        self.validation_warnings = []
        self.suggestions = []
        
    def validate_dimensions(self, config_dict: Dict[str, Any], 
                          data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """验证维度配置"""
        validation_results = {'errors': [], 'warnings': [], 'suggestions': []}
        
        # 检查必要的参数列表
        required_params = ['STRUCT_PARAMS', 'SPECTRA_PARAMS', 'METRIC_PARAMS']
        for param in required_params:
            if param not in config_dict:
                validation_results['errors'].append(f"缺少必要的参数配置: {param}")
        
        if data is not None:
            # 验证参数是否存在于数据中
            for param_type in required_params:
                if param_type in config_dict:
                    param_list = config_dict[param_type]
                    missing_cols = [col for col in param_list if col not in data.columns]
                    if missing_cols:
                        validation_results['errors'].append(
                            f"{param_type} 中的列不存在于数据中: {missing_cols}"
                        )
        
        # 检查维度一致性
        if all(param in config_dict for param in required_params):
            struct_dim = len(config_dict['STRUCT_PARAMS'])
            spectra_dim = len(config_dict['SPECTRA_PARAMS'])
            metric_dim = len(config_dict['METRIC_PARAMS'])
            
            # 检查条件维度设置
            condition_dim = config_dict.get('CONDITION_DIM', 0)
            if condition_dim != metric_dim:
                validation_results['warnings'].append(
                    f"CONDITION_DIM ({condition_dim}) 与 METRIC_PARAMS 长度 ({metric_dim}) 不匹配"
                )
        
        return validation_results
    
    def validate_training_params(self, config_dict: Dict[str, Any], 
                               total_samples: int) -> Dict[str, Any]:
        """验证训练参数"""
        validation_results = {'errors': [], 'warnings': [], 'suggestions': []}
        
        # 批次大小检查
        batch_size = config_dict.get('BATCH_SIZE', 64)
        if batch_size > total_samples // 4:
            validation_results['warnings'].append(
                f"批次大小 ({batch_size}) 相对于数据集大小 ({total_samples}) 过大"
            )
            suggested_batch_size = max(8, total_samples // 16)
            validation_results['suggestions'].append(
                f"建议批次大小: {suggested_batch_size}"
            )
        
        # 数据分割检查
        test_split = config_dict.get('TEST_SPLIT', 0.1)
        val_split = config_dict.get('VAL_SPLIT', 0.1)
        
        if test_split + val_split >= 0.5:
            validation_results['warnings'].append(
                f"测试集和验证集比例过大: test={test_split}, val={val_split}"
            )
        
        train_samples = int(total_samples * (1 - test_split - val_split))
        if train_samples < 1000:
            validation_results['warnings'].append(
                f"训练样本数量过少: {train_samples}"
            )
            validation_results['suggestions'].append("考虑使用交叉验证或减少验证集比例")
        
        # 学习率检查
        lr = config_dict.get('LR', 1e-4)
        if lr > 1e-2:
            validation_results['warnings'].append(f"学习率过大: {lr}")
        elif lr < 1e-6:
            validation_results['warnings'].append(f"学习率过小: {lr}")
        
        return validation_results
    
    def validate_loss_weights(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """验证损失权重配置"""
        validation_results = {'errors': [], 'warnings': [], 'suggestions': []}
        
        # 检查权重平衡
        spectra_weight = config_dict.get('SPECTRA_LOSS_WEIGHT', 0.01)
        metric_weight = config_dict.get('METRIC_LOSS_WEIGHT', 1.0)
        
        if spectra_weight / metric_weight > 10 or metric_weight / spectra_weight > 10:
            validation_results['warnings'].append(
                f"损失权重可能不平衡: spectra={spectra_weight}, metric={metric_weight}"
            )
        
        # 检查物理损失权重
        physics_weight = config_dict.get('LAMBDA_PHYSICS', 0.1)
        if physics_weight > 1.0:
            validation_results['warnings'].append(
                f"物理损失权重过大: {physics_weight}"
            )
        elif physics_weight < 0.01:
            validation_results['warnings'].append(
                f"物理损失权重过小: {physics_weight}"
            )
        
        return validation_results
    
    def validate_device_config(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        """验证设备配置"""
        validation_results = {'errors': [], 'warnings': [], 'suggestions': []}
        
        device = config_dict.get('DEVICE', 'cpu')
        
        if device == 'cuda':
            if not torch.cuda.is_available():
                validation_results['errors'].append("CUDA不可用，但配置为使用GPU")
                validation_results['suggestions'].append("将DEVICE设置为'cpu'")
            else:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory < 4 * 1024**3:  # 小于4GB
                    validation_results['warnings'].append(
                        f"GPU内存较小: {gpu_memory / 1024**3:.1f}GB"
                    )
                    validation_results['suggestions'].append("考虑减小批次大小")
        
        return validation_results

class ConfigOptimizer:
    """配置优化器"""
    
    def __init__(self, total_samples: int):
        self.total_samples = total_samples
        
    def suggest_optimal_config(self, current_config: Dict[str, Any],
                             boundaries: Dict[str, BoundaryInfo]) -> Dict[str, Any]:
        """建议优化的配置"""
        optimized_config = current_config.copy()
        
        # 优化批次大小
        if self.total_samples < 1000:
            optimized_config['BATCH_SIZE'] = min(16, self.total_samples // 8)
        elif self.total_samples < 5000:
            optimized_config['BATCH_SIZE'] = min(32, self.total_samples // 16)
        else:
            optimized_config['BATCH_SIZE'] = min(64, self.total_samples // 32)
        
        # 优化数据分割
        if self.total_samples < 2000:
            optimized_config['TEST_SPLIT'] = 0.15
            optimized_config['VAL_SPLIT'] = 0.15
        else:
            optimized_config['TEST_SPLIT'] = 0.1
            optimized_config['VAL_SPLIT'] = 0.1
        
        # 优化边界设置
        if boundaries:
            struct_min_bounds = []
            struct_max_bounds = []
            
            for param in current_config.get('STRUCT_PARAMS', []):
                if param in boundaries:
                    boundary_info = boundaries[param]
                    struct_min_bounds.append(boundary_info.recommended_min)
                    struct_max_bounds.append(boundary_info.recommended_max)
                else:
                    # 保持原始值或使用默认值
                    struct_min_bounds.append(0.0)
                    struct_max_bounds.append(1.0)
            
            optimized_config['STRUCT_MIN_BOUNDS'] = struct_min_bounds
            optimized_config['STRUCT_MAX_BOUNDS'] = struct_max_bounds
        
        # 优化训练参数（针对小数据集）
        if self.total_samples < 5000:
            optimized_config['EPOCHS'] = min(800, current_config.get('EPOCHS', 500))
            optimized_config['LR'] = max(5e-5, current_config.get('LR', 1e-4) * 0.5)
            optimized_config['PRETRAIN_FWD_MODEL_EPOCHS'] = min(800, 
                current_config.get('PRETRAIN_FWD_MODEL_EPOCHS', 500))
        
        return optimized_config

def auto_detect_and_update_config(dataset_path: str, current_config: Dict[str, Any],
                                 save_path: Optional[str] = None) -> Dict[str, Any]:
    """自动检测并更新配置"""
    logger.info("开始自动配置检测和更新...")
    
    # 加载数据
    df = pd.read_csv(dataset_path)
    logger.info(f"加载数据集: {len(df)} 样本")
    
    # 边界检测
    detector = AutoBoundaryDetector()
    struct_params = current_config.get('STRUCT_PARAMS', ['r1', 'r2', 'w', 'g'])
    boundaries = detector.detect_boundaries(df, struct_params)
    
    # 配置验证
    validator = ConfigValidator()
    
    dim_validation = validator.validate_dimensions(current_config, df)
    train_validation = validator.validate_training_params(current_config, len(df))
    loss_validation = validator.validate_loss_weights(current_config)
    device_validation = validator.validate_device_config(current_config)
    
    # 边界验证
    current_bounds = {}
    struct_min = current_config.get('STRUCT_MIN_BOUNDS', [])
    struct_max = current_config.get('STRUCT_MAX_BOUNDS', [])
    
    if len(struct_min) == len(struct_params) and len(struct_max) == len(struct_params):
        for i, param in enumerate(struct_params):
            current_bounds[param] = (struct_min[i], struct_max[i])
    
    boundary_validation = detector.validate_existing_boundaries(boundaries, current_bounds)
    
    # 配置优化
    optimizer = ConfigOptimizer(len(df))
    optimized_config = optimizer.suggest_optimal_config(current_config, boundaries)
    
    # 汇总结果
    results = {
        'original_config': current_config,
        'optimized_config': optimized_config,
        'boundaries': boundaries,
        'validations': {
            'dimensions': dim_validation,
            'training': train_validation,
            'loss_weights': loss_validation,
            'device': device_validation,
            'boundaries': boundary_validation
        },
        'total_samples': len(df)
    }
    
    # 保存结果
    if save_path:
        with open(save_path, 'w', encoding='utf-8') as f:
            # 将BoundaryInfo对象转换为字典以便JSON序列化
            serializable_results = results.copy()
            serializable_boundaries = {}
            for param, boundary_info in boundaries.items():
                serializable_boundaries[param] = {
                    'min_value': boundary_info.min_value,
                    'max_value': boundary_info.max_value,
                    'range_value': boundary_info.range_value,
                    'mean_value': boundary_info.mean_value,
                    'std_value': boundary_info.std_value,
                    'percentile_5': boundary_info.percentile_5,
                    'percentile_95': boundary_info.percentile_95,
                    'recommended_min': boundary_info.recommended_min,
                    'recommended_max': boundary_info.recommended_max
                }
            serializable_results['boundaries'] = serializable_boundaries
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        logger.info(f"配置分析结果保存到: {save_path}")
    
    return results

if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from config import config
    
    print("=== 配置验证和边界检测测试 ===")
    
    # 获取数据集路径
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    dataset_path = os.path.join(project_root, config.DATASET_PATH)
    
    # 当前配置
    current_config = {
        'STRUCT_PARAMS': config.STRUCT_PARAMS,
        'SPECTRA_PARAMS': config.SPECTRA_PARAMS,
        'METRIC_PARAMS': config.METRIC_PARAMS,
        'BATCH_SIZE': config.BATCH_SIZE,
        'TEST_SPLIT': config.TEST_SPLIT,
        'VAL_SPLIT': config.VAL_SPLIT,
        'LR': config.LR,
        'DEVICE': config.DEVICE,
        'STRUCT_MIN_BOUNDS': config.STRUCT_MIN_BOUNDS,
        'STRUCT_MAX_BOUNDS': config.STRUCT_MAX_BOUNDS,
        'CONDITION_DIM': config.CONDITION_DIM,
        'SPECTRA_LOSS_WEIGHT': config.SPECTRA_LOSS_WEIGHT,
        'METRIC_LOSS_WEIGHT': config.METRIC_LOSS_WEIGHT,
        'LAMBDA_PHYSICS': config.LAMBDA_PHYSICS,
        'EPOCHS': config.EPOCHS,
        'PRETRAIN_FWD_MODEL_EPOCHS': config.PRETRAIN_FWD_MODEL_EPOCHS,
    }
    
    # 执行自动检测和更新
    try:
        results = auto_detect_and_update_config(
            dataset_path, 
            current_config,
            save_path=os.path.join(project_root, 'config_analysis_results.json')
        )
        
        print(f"数据集大小: {results['total_samples']} 样本")
        print(f"检测到边界: {len(results['boundaries'])} 个参数")
        
        # 打印主要建议
        optimized = results['optimized_config']
        original = results['original_config']
        
        print("\n=== 主要配置建议 ===")
        if optimized['BATCH_SIZE'] != original['BATCH_SIZE']:
            print(f"批次大小: {original['BATCH_SIZE']} → {optimized['BATCH_SIZE']}")
        
        if optimized['STRUCT_MIN_BOUNDS'] != original['STRUCT_MIN_BOUNDS']:
            print(f"结构参数下界更新")
        
        if optimized['STRUCT_MAX_BOUNDS'] != original['STRUCT_MAX_BOUNDS']:
            print(f"结构参数上界更新")
        
        print("配置验证和边界检测完成！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        print("这可能是由于数据文件不存在或环境问题导致的")
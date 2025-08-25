"""
数据质量检查和处理模块
专门为小数据集（4450条）优化的数据处理工具
"""

import pandas as pd
import numpy as np
import torch
import warnings
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import KFold
import logging

class DataQualityChecker:
    """数据质量检查器，专门针对小数据集设计"""
    
    def __init__(self, small_dataset_threshold: int = 10000):
        self.small_dataset_threshold = small_dataset_threshold
        self.logger = logging.getLogger(__name__)
        
    def check_data_integrity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """全面的数据完整性检查"""
        results = {}
        
        # 基本信息
        results['total_samples'] = len(df)
        results['total_features'] = len(df.columns)
        results['is_small_dataset'] = len(df) < self.small_dataset_threshold
        
        # 缺失值检查
        missing_info = self._check_missing_values(df)
        results['missing_values'] = missing_info
        
        # 异常值检查
        outlier_info = self._check_outliers(df)
        results['outliers'] = outlier_info
        
        # 数据分布检查
        distribution_info = self._check_distributions(df)
        results['distributions'] = distribution_info
        
        # 重复值检查
        duplicates_info = self._check_duplicates(df)
        results['duplicates'] = duplicates_info
        
        # 数据范围检查
        range_info = self._check_data_ranges(df)
        results['data_ranges'] = range_info
        
        return results
    
    def _check_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查缺失值"""
        missing_counts = df.isnull().sum()
        missing_ratios = missing_counts / len(df)
        
        critical_missing = missing_ratios[missing_ratios > 0.05]  # 小数据集阈值降低
        
        info = {
            'total_missing': missing_counts.sum(),
            'missing_by_column': missing_counts[missing_counts > 0].to_dict(),
            'missing_ratios': missing_ratios[missing_ratios > 0].to_dict(),
            'critical_columns': critical_missing.index.tolist() if len(critical_missing) > 0 else []
        }
        
        if len(critical_missing) > 0:
            warnings.warn(f"发现严重缺失值：{critical_missing.to_dict()}")
            
        return info
    
    def _check_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查异常值（使用IQR方法和Z-score）"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            # IQR方法
            Q1, Q3 = data.quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            iqr_outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            
            # Z-score方法（修正版，对小数据集更宽松）
            z_scores = np.abs(stats.zscore(data))
            z_threshold = 3.0  # 小数据集使用更宽松的阈值
            zscore_outliers = (z_scores > z_threshold).sum()
            
            outlier_info[col] = {
                'iqr_outliers': iqr_outliers,
                'iqr_ratio': iqr_outliers / len(data),
                'zscore_outliers': zscore_outliers,
                'zscore_ratio': zscore_outliers / len(data),
                'bounds': {'lower': lower_bound, 'upper': upper_bound}
            }
            
            # 小数据集的异常值警告阈值调整
            if iqr_outliers > len(data) * 0.1:  # 10%阈值
                warnings.warn(f"列 {col} 异常值比例过高: {iqr_outliers/len(data):.2%}")
                
        return outlier_info
    
    def _check_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据分布"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        distribution_info = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            
            # 基本统计
            stats_info = {
                'mean': data.mean(),
                'std': data.std(),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min()
            }
            
            # 正态性检验（对小数据集使用Shapiro-Wilk）
            if len(data) >= 3:
                normality_stat, normality_p = stats.shapiro(data) if len(data) <= 5000 else stats.normaltest(data)
                stats_info['normality_test'] = {
                    'statistic': normality_stat,
                    'p_value': normality_p,
                    'is_normal': normality_p > 0.05
                }
            
            distribution_info[col] = stats_info
            
        return distribution_info
    
    def _check_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查重复值"""
        total_duplicates = df.duplicated().sum()
        duplicate_ratio = total_duplicates / len(df)
        
        info = {
            'total_duplicates': total_duplicates,
            'duplicate_ratio': duplicate_ratio,
            'unique_samples': len(df) - total_duplicates
        }
        
        # 小数据集中重复值更关键
        if duplicate_ratio > 0.01:  # 1%阈值
            warnings.warn(f"发现重复样本：{total_duplicates}条 ({duplicate_ratio:.2%})")
            
        return info
    
    def _check_data_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """检查数据范围的合理性"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        range_info = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            range_info[col] = {
                'min': data.min(),
                'max': data.max(),
                'range': data.max() - data.min(),
                'zero_values': (data == 0).sum(),
                'negative_values': (data < 0).sum()
            }
            
        return range_info

class SmallDatasetOptimizer:
    """专门为小数据集设计的优化器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def suggest_split_strategy(self, total_samples: int) -> Dict[str, float]:
        """为小数据集建议分割策略"""
        if total_samples < 1000:
            # 超小数据集：更多用于训练
            return {'train': 0.8, 'val': 0.1, 'test': 0.1}
        elif total_samples < 5000:
            # 小数据集（当前情况）：平衡训练和验证
            return {'train': 0.7, 'val': 0.15, 'test': 0.15}
        else:
            # 常规分割
            return {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    def suggest_batch_size(self, total_samples: int, train_ratio: float = 0.7) -> int:
        """为小数据集建议批次大小"""
        train_samples = int(total_samples * train_ratio)
        
        # 确保有足够的批次进行训练
        if train_samples < 100:
            return min(8, train_samples // 4)
        elif train_samples < 500:
            return min(16, train_samples // 8)
        elif train_samples < 2000:
            return min(32, train_samples // 16)
        else:
            return min(64, train_samples // 32)
    
    def suggest_cross_validation(self, total_samples: int) -> Dict[str, Any]:
        """建议交叉验证策略"""
        if total_samples < 1000:
            return {
                'use_cv': True,
                'n_folds': 5,
                'strategy': 'stratified_kfold',
                'reason': '超小数据集建议使用5折交叉验证'
            }
        elif total_samples < 5000:
            return {
                'use_cv': True,
                'n_folds': 10,
                'strategy': 'kfold',
                'reason': '小数据集建议使用10折交叉验证'
            }
        else:
            return {
                'use_cv': False,
                'n_folds': None,
                'strategy': 'holdout',
                'reason': '数据集足够大，使用留出法'
            }

class NaNHandler:
    """统一的NaN处理器"""
    
    def __init__(self, strategy: str = 'adaptive'):
        """
        策略选项:
        - 'adaptive': 根据缺失比例自适应选择
        - 'drop': 删除含有NaN的行
        - 'interpolate': 插值填充
        - 'mean': 均值填充
        - 'median': 中位数填充
        - 'forward_fill': 前向填充
        """
        self.strategy = strategy
        self.logger = logging.getLogger(__name__)
        
    def handle_nans(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """处理NaN值"""
        original_shape = df.shape
        nan_info = {}
        
        # 记录原始NaN情况
        nan_counts = df.isnull().sum()
        nan_columns = nan_counts[nan_counts > 0].index.tolist()
        
        if len(nan_columns) == 0:
            return df.copy(), {'status': 'no_nans_found', 'original_shape': original_shape}
        
        df_cleaned = df.copy()
        
        for col in nan_columns:
            nan_count = nan_counts[col]
            nan_ratio = nan_count / len(df)
            
            if self.strategy == 'adaptive':
                if nan_ratio > 0.3:
                    # 缺失比例过高，删除列
                    df_cleaned = df_cleaned.drop(columns=[col])
                    self.logger.warning(f"删除列 {col}，缺失比例过高: {nan_ratio:.2%}")
                elif nan_ratio > 0.1:
                    # 中等缺失，使用插值
                    if df_cleaned[col].dtype in ['float64', 'int64']:
                        df_cleaned[col] = df_cleaned[col].interpolate()
                        # 如果首尾还有NaN，用均值填充
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # 少量缺失，使用均值/中位数
                    if df_cleaned[col].dtype in ['float64', 'int64']:
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
                    else:
                        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'unknown')
            
            elif self.strategy == 'drop':
                df_cleaned = df_cleaned.dropna()
                
            elif self.strategy == 'interpolate':
                if df_cleaned[col].dtype in ['float64', 'int64']:
                    df_cleaned[col] = df_cleaned[col].interpolate()
                    df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mean())
                    
            # 其他策略的实现...
        
        # 最终检查
        final_nan_count = df_cleaned.isnull().sum().sum()
        
        nan_info = {
            'status': 'processed',
            'original_shape': original_shape,
            'final_shape': df_cleaned.shape,
            'original_nan_count': nan_counts.sum(),
            'final_nan_count': final_nan_count,
            'processed_columns': nan_columns,
            'strategy_used': self.strategy
        }
        
        return df_cleaned, nan_info

def validate_small_dataset_setup(df: pd.DataFrame, config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """验证小数据集的设置是否合理"""
    total_samples = len(df)
    checker = DataQualityChecker()
    optimizer = SmallDatasetOptimizer()
    
    # 数据质量检查
    quality_report = checker.check_data_integrity(df)
    
    # 优化建议
    split_suggestion = optimizer.suggest_split_strategy(total_samples)
    batch_suggestion = optimizer.suggest_batch_size(total_samples)
    cv_suggestion = optimizer.suggest_cross_validation(total_samples)
    
    # 配置验证
    current_test_split = config_dict.get('TEST_SPLIT', 0.1)
    current_val_split = config_dict.get('VAL_SPLIT', 0.1)
    current_batch_size = config_dict.get('BATCH_SIZE', 64)
    
    validation_results = {
        'data_quality': quality_report,
        'current_config': {
            'test_split': current_test_split,
            'val_split': current_val_split,
            'batch_size': current_batch_size,
            'total_samples': total_samples
        },
        'suggestions': {
            'split_strategy': split_suggestion,
            'batch_size': batch_suggestion,
            'cross_validation': cv_suggestion
        },
        'warnings': [],
        'recommendations': []
    }
    
    # 生成警告和建议
    if current_batch_size > batch_suggestion:
        validation_results['warnings'].append(f"批次大小过大：当前{current_batch_size}，建议{batch_suggestion}")
        
    if total_samples < 5000:
        validation_results['recommendations'].append("考虑使用数据增强技术")
        validation_results['recommendations'].append("使用更强的正则化")
        validation_results['recommendations'].append("考虑迁移学习或预训练模型")
        
    if cv_suggestion['use_cv']:
        validation_results['recommendations'].append(f"建议使用{cv_suggestion['n_folds']}折交叉验证")
    
    return validation_results

if __name__ == '__main__':
    # 测试代码
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from config import config
    
    # 加载数据
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    dataset_path = os.path.join(project_root, config.DATASET_PATH)
    df = pd.read_csv(dataset_path)
    
    print("=== 数据质量检查报告 ===")
    
    # 数据质量检查
    checker = DataQualityChecker()
    quality_report = checker.check_data_integrity(df)
    
    print(f"数据集大小: {quality_report['total_samples']} 样本")
    print(f"特征数量: {quality_report['total_features']}")
    print(f"是否为小数据集: {quality_report['is_small_dataset']}")
    
    # NaN处理
    nan_handler = NaNHandler(strategy='adaptive')
    df_cleaned, nan_info = nan_handler.handle_nans(df)
    print(f"\nNaN处理结果: {nan_info['status']}")
    
    # 配置验证
    config_dict = {
        'TEST_SPLIT': config.TEST_SPLIT,
        'VAL_SPLIT': config.VAL_SPLIT,
        'BATCH_SIZE': config.BATCH_SIZE
    }
    
    validation_results = validate_small_dataset_setup(df_cleaned, config_dict)
    
    print("\n=== 配置建议 ===")
    print(f"建议分割策略: {validation_results['suggestions']['split_strategy']}")
    print(f"建议批次大小: {validation_results['suggestions']['batch_size']}")
    print(f"交叉验证建议: {validation_results['suggestions']['cross_validation']}")
    
    if validation_results['warnings']:
        print("\n⚠️ 警告:")
        for warning in validation_results['warnings']:
            print(f"  - {warning}")
    
    if validation_results['recommendations']:
        print("\n💡 建议:")
        for rec in validation_results['recommendations']:
            print(f"  - {rec}")
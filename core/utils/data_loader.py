

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
import sys
import os
import warnings
import logging
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import config
from .data_quality import DataQualityChecker, NaNHandler, validate_small_dataset_setup

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetamaterialDataset(Dataset):
    """Custom PyTorch Dataset for the Metamaterial data."""
    def __init__(self, struct_params, spectra, metrics):
        self.struct_params = torch.FloatTensor(struct_params)
        self.spectra = torch.FloatTensor(spectra)
        self.metrics = torch.FloatTensor(metrics)

    def __len__(self):
        return len(self.struct_params)

    def __getitem__(self, idx):
        return {
            'struct': self.struct_params[idx],
            'spectra': self.spectra[idx],
            'metrics': self.metrics[idx]
        }

def get_dataloaders(batch_size=config.BATCH_SIZE, enable_quality_check=True, scaler_type='minmax'):
    """
    加载数据，进行质量检查和优化，分割数据集并返回DataLoaders和scalers
    
    Args:
        batch_size: 批次大小，如果为None则自动推荐
        enable_quality_check: 是否启用数据质量检查
        scaler_type: 缩放器类型 ('minmax', 'robust')
    """
    # Load the dataset
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    dataset_full_path = os.path.join(project_root, config.DATASET_PATH)
    
    logger.info(f"加载数据集: {dataset_full_path}")
    df = pd.read_csv(dataset_full_path)
    
    # 数据质量检查和处理
    if enable_quality_check:
        logger.info("执行数据质量检查...")
        
        # NaN处理
        nan_handler = NaNHandler(strategy='adaptive')
        df, nan_info = nan_handler.handle_nans(df)
        logger.info(f"NaN处理完成: {nan_info['status']}")
        
        # 配置验证
        config_dict = {
            'TEST_SPLIT': config.TEST_SPLIT,
            'VAL_SPLIT': config.VAL_SPLIT,
            'BATCH_SIZE': batch_size or config.BATCH_SIZE
        }
        
        validation_results = validate_small_dataset_setup(df, config_dict)
        
        # 如果未指定批次大小，使用建议值
        if batch_size is None:
            batch_size = validation_results['suggestions']['batch_size']
            logger.info(f"使用建议的批次大小: {batch_size}")
        
        # 打印警告
        for warning in validation_results['warnings']:
            logger.warning(warning)
        
        # 打印建议
        for rec in validation_results['recommendations']:
            logger.info(f"建议: {rec}")
    
    # 检查数据维度匹配
    _validate_data_dimensions(df)

    # Separate raw features (no scaling yet)
    struct_params = df[config.STRUCT_PARAMS].values
    spectra = df[config.SPECTRA_PARAMS].values
    metrics = df[config.METRIC_PARAMS].values

    # First split off test set
    struct_trainval, struct_test, spectra_trainval, spectra_test, metrics_trainval, metrics_test = train_test_split(
        struct_params, spectra, metrics,
        test_size=config.TEST_SPLIT,
        random_state=config.RANDOM_STATE,
        shuffle=True
    )

    # Then split train/val from trainval
    val_relative = config.VAL_SPLIT / (1.0 - config.TEST_SPLIT) if (1.0 - config.TEST_SPLIT) > 0 else 0.0
    struct_train, struct_val, spectra_train, spectra_val, metrics_train, metrics_val = train_test_split(
        struct_trainval, spectra_trainval, metrics_trainval,
        test_size=val_relative,
        random_state=config.RANDOM_STATE,
        shuffle=True
    )

    # 选择缩放器类型（小数据集推荐RobustScaler）
    if scaler_type == 'robust':
        scaler_struct = RobustScaler().fit(struct_train)
        scaler_spectra = RobustScaler().fit(spectra_train)
        scaler_metrics = RobustScaler().fit(metrics_train)
        logger.info("使用RobustScaler（推荐用于小数据集）")
    else:
        scaler_struct = MinMaxScaler().fit(struct_train)
        scaler_spectra = MinMaxScaler().fit(spectra_train)
        scaler_metrics = MinMaxScaler().fit(metrics_train)
        logger.info("使用MinMaxScaler")
    
    # 检查缩放后的数据质量
    _validate_scaled_data(struct_train_scaled, "struct_train")
    _validate_scaled_data(spectra_train_scaled, "spectra_train")
    _validate_scaled_data(metrics_train_scaled, "metrics_train")

    # Transform each split
    struct_train_scaled = scaler_struct.transform(struct_train)
    struct_val_scaled = scaler_struct.transform(struct_val)
    struct_test_scaled = scaler_struct.transform(struct_test)

    spectra_train_scaled = scaler_spectra.transform(spectra_train)
    spectra_val_scaled = scaler_spectra.transform(spectra_val)
    spectra_test_scaled = scaler_spectra.transform(spectra_test)

    metrics_train_scaled = scaler_metrics.transform(metrics_train)
    metrics_val_scaled = scaler_metrics.transform(metrics_val)
    metrics_test_scaled = scaler_metrics.transform(metrics_test)

    # Create datasets per split
    train_dataset = MetamaterialDataset(struct_train_scaled, spectra_train_scaled, metrics_train_scaled)
    val_dataset = MetamaterialDataset(struct_val_scaled, spectra_val_scaled, metrics_val_scaled)
    test_dataset = MetamaterialDataset(struct_test_scaled, spectra_test_scaled, metrics_test_scaled)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Return loaders and scalers (scalers needed for inverse transform)
    scalers = {
        'struct': scaler_struct,
        'spectra': scaler_spectra,
        'metrics': scaler_metrics
    }

    return train_loader, val_loader, test_loader, scalers

def _validate_data_dimensions(df: pd.DataFrame):
    """验证数据维度是否与配置匹配"""
    missing_struct = [col for col in config.STRUCT_PARAMS if col not in df.columns]
    missing_spectra = [col for col in config.SPECTRA_PARAMS if col not in df.columns]
    missing_metrics = [col for col in config.METRIC_PARAMS if col not in df.columns]
    
    if missing_struct:
        raise ValueError(f"缺少结构参数列: {missing_struct}")
    if missing_spectra:
        raise ValueError(f"缺少光谱参数列: {missing_spectra}")
    if missing_metrics:
        raise ValueError(f"缺少指标参数列: {missing_metrics}")

def _validate_scaled_data(data: np.ndarray, name: str):
    """验证缩放后的数据质量"""
    if np.isnan(data).any():
        warnings.warn(f"{name} 中存在NaN值")
    
    if np.isinf(data).any():
        warnings.warn(f"{name} 中存在无穷大值")
    
    # 检查是否有异常的数值范围
    data_min, data_max = data.min(), data.max()
    if data_max - data_min < 1e-8:
        warnings.warn(f"{name} 数据范围过小，可能存在问题")

def get_cross_validation_dataloaders(n_folds=5, batch_size=None):
    """
    为小数据集提供交叉验证数据加载器
    
    Returns:
        Generator yielding (train_loader, val_loader, scalers) for each fold
    """
    # 加载和处理数据
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    dataset_full_path = os.path.join(project_root, config.DATASET_PATH)
    df = pd.read_csv(dataset_full_path)
    
    # 数据预处理
    nan_handler = NaNHandler(strategy='adaptive')
    df, _ = nan_handler.handle_nans(df)
    
    # 准备数据
    struct_params = df[config.STRUCT_PARAMS].values
    spectra = df[config.SPECTRA_PARAMS].values
    metrics = df[config.METRIC_PARAMS].values
    
    if batch_size is None:
        from .data_quality import SmallDatasetOptimizer
        optimizer = SmallDatasetOptimizer()
        batch_size = optimizer.suggest_batch_size(len(df))
    
    # K折交叉验证
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_STATE)
    
    # 使用第一个指标作为分层依据（简化处理）
    stratify_labels = pd.cut(metrics[:, 0], bins=5, labels=False)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(struct_params, stratify_labels)):
        logger.info(f"处理第 {fold+1}/{n_folds} 折")
        
        # 分割数据
        struct_train = struct_params[train_idx]
        struct_val = struct_params[val_idx]
        spectra_train = spectra[train_idx]
        spectra_val = spectra[val_idx]
        metrics_train = metrics[train_idx]
        metrics_val = metrics[val_idx]
        
        # 拟合缩放器
        scaler_struct = RobustScaler().fit(struct_train)
        scaler_spectra = RobustScaler().fit(spectra_train)
        scaler_metrics = RobustScaler().fit(metrics_train)
        
        # 缩放数据
        struct_train_scaled = scaler_struct.transform(struct_train)
        struct_val_scaled = scaler_struct.transform(struct_val)
        spectra_train_scaled = scaler_spectra.transform(spectra_train)
        spectra_val_scaled = scaler_spectra.transform(spectra_val)
        metrics_train_scaled = scaler_metrics.transform(metrics_train)
        metrics_val_scaled = scaler_metrics.transform(metrics_val)
        
        # 创建数据集
        train_dataset = MetamaterialDataset(struct_train_scaled, spectra_train_scaled, metrics_train_scaled)
        val_dataset = MetamaterialDataset(struct_val_scaled, spectra_val_scaled, metrics_val_scaled)
        
        # 创建DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        scalers = {
            'struct': scaler_struct,
            'spectra': scaler_spectra,
            'metrics': scaler_metrics
        }
        
        yield train_loader, val_loader, scalers

if __name__ == '__main__':
    # Example of how to use the dataloader
    train_loader, val_loader, test_loader, scalers = get_dataloaders()

    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of testing batches: {len(test_loader)}")

    # Inspect a batch
    sample_batch = next(iter(train_loader))
    print("\nSample batch shapes:")
    print(f"  Structure params: {sample_batch['struct'].shape}")
    print(f"  Spectra:          {sample_batch['spectra'].shape}")
    print(f"  Metrics:          {sample_batch['metrics'].shape}")

    # Check scaler functionality
    original_struct_sample = scalers['struct'].inverse_transform(sample_batch['struct'].numpy())
    print("\nExample of inverse-transformed structure parameters:")
    print(original_struct_sample[0])

    # Add these lines to print statistics of scaled data
    print("\n--- Scaled Data Statistics (Train Set) ---")
    train_struct_data = torch.cat([batch['struct'] for batch in train_loader])
    train_spectra_data = torch.cat([batch['spectra'] for batch in train_loader])
    train_metrics_data = torch.cat([batch['metrics'] for batch in train_loader])

    print(f"Scaled Struct - Min: {train_struct_data.min():.4f}, Max: {train_struct_data.max():.4f}, Mean: {train_struct_data.mean():.4f}, Std: {train_struct_data.std():.4f}")
    print(f"Scaled Spectra - Min: {train_spectra_data.min():.4f}, Max: {train_spectra_data.max():.4f}, Mean: {train_spectra_data.mean():.4f}, Std: {train_spectra_data.std():.4f}")
    print(f"Scaled Metrics - Min: {train_metrics_data.min():.4f}, Max: {train_metrics_data.max():.4f}, Mean: {train_metrics_data.mean():.4f}, Std: {train_metrics_data.std():.4f}")

    print("\nScaler Scale_ values (for original range calculation):")
    print(f"Spectra Scale_: {scalers['spectra'].scale_}")
    print(f"Metrics Scale_: {scalers['metrics'].scale_}")


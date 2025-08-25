import torch
import os
import sys

# 添加工具模块路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- General Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "dataset/THz_Metamaterial_Spectra_With_Metrics.csv"
CHECKPOINT_DIR = "checkpoints"
SAVED_MODELS_DIR = "saved_models"
LOG_DIR = "logs"
GENERATED_DATA_DIR = "generated_data"
PLOT_DIR="plots"

# 自动配置标志
AUTO_DETECT_BOUNDARIES = True  # 是否自动检测边界
VALIDATE_CONFIG = True         # 是否验证配置
# --- Forward Model Options ---
# Choose spectra activation: 'sigmoid' or 'tanh'
SPECTRA_ACTIVATION = 'sigmoid'
# Choose metric extractor: 'trainable' or 'physics'
METRIC_EXTRACTOR_TYPE = 'trainable'

# --- Data Preprocessing ---
STRUCT_PARAMS = ['r1', 'r2', 'w', 'g']
SPECTRA_PARAMS = [f'Freq_{i/100:.2f}' for i in range(50, 300)]
METRIC_PARAMS = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
RANDOM_STATE = 42

# Structural parameter bounds for PhysicalConstraintModule in Discriminator
# These values should reflect the normalized range of your structural parameters
# For example, if r1, r2, w, g are normalized to [0, 1], then use [0.0, 0.0, 0.0, 0.0] and [1.0, 1.0, 1.0, 1.0]
# If they are in a specific physical range, use that range after normalization.
# Based on your previous output, the original ranges were roughly r1,r2,w: [2.2, 2.8], g: [1.8, 3.0]
# Assuming data is normalized to [0,1], these bounds should also be normalized.
# For now, I'll use a generic [0,1] range. PLEASE ADJUST THESE BASED ON YOUR ACTUAL DATA NORMALIZATION.
STRUCT_MIN_BOUNDS = [0.0, 0.0, 0.0, 0.0] 
STRUCT_MAX_BOUNDS = [1.0, 1.0, 1.0, 1.0] 

# Relationship constraints among structural parameters (applied in Generator projector)
# If your domain requires r1 >= r2 (outer radius >= inner radius), keep True.
# Otherwise set to False.
ENFORCE_R1_GE_R2 = True

# --- Training Hyperparameters ---
# General
EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-4

# Forward Model Pre-training
PRETRAIN_FWD_MODEL_EPOCHS = 500
PRETRAIN_FWD_MODEL_LR = 5e-4
SPECTRA_LOSS_WEIGHT = 0.01  # Weight for spectra prediction loss
METRIC_LOSS_WEIGHT = 1.0   # Weight for metric prediction loss

# PI-GAN Training
GAN_EPOCHS = 1000
G_LR = 5e-5
D_LR = 5e-5
LATENT_DIM = 100

# Conditional Generator Setting
# If conditioning on METRIC_PARAMS, then CONDITION_DIM = len(METRIC_PARAMS)
CONDITION_DIM = len(METRIC_PARAMS) # Assuming conditioning on all metrics

# Loss weights
LAMBDA_GP = 10  # Gradient Penalty
LAMBDA_R1 = 0.0  # R1 regularization weight (gamma/2). Set >0 to enable.
LAMBDA_PHYSICS = 0.1 # Physics-informed loss
LAMBDA_METRIC = 0.05 # Metric loss
LAMBDA_PID_FEEDBACK = 0.01 # Enable small weight for physical feedback initially
PID_FEEDBACK_ANNEAL_EPOCHS = 200 # Linearly anneal PID weight from initial to target over these GAN epochs
PID_FEEDBACK_TARGET = 0.05 # Target PID weight at the end of annealing

# Diversity enhancement (Mode-Seeking)
LAMBDA_MODE_SEEKING = 0.1  # Weight for mode-seeking loss encouraging diverse outputs per condition
MODE_SEEKING_EPS = 1e-6    # Small epsilon to avoid division by zero

# === 自动配置检测和验证 ===
def validate_and_update_config():
    """自动验证和更新配置"""
    if not (AUTO_DETECT_BOUNDARIES or VALIDATE_CONFIG):
        return
    
    try:
        # 导入验证模块
        from core.utils.config_validator import auto_detect_and_update_config
        import pandas as pd
        
        # 获取数据集路径
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        dataset_full_path = os.path.join(project_root, DATASET_PATH)
        
        if not os.path.exists(dataset_full_path):
            print(f"警告: 数据集文件不存在: {dataset_full_path}")
            return
        
        # 当前配置
        current_config = {
            'STRUCT_PARAMS': STRUCT_PARAMS,
            'SPECTRA_PARAMS': SPECTRA_PARAMS,
            'METRIC_PARAMS': METRIC_PARAMS,
            'BATCH_SIZE': BATCH_SIZE,
            'TEST_SPLIT': TEST_SPLIT,
            'VAL_SPLIT': VAL_SPLIT,
            'LR': LR,
            'DEVICE': DEVICE,
            'STRUCT_MIN_BOUNDS': STRUCT_MIN_BOUNDS,
            'STRUCT_MAX_BOUNDS': STRUCT_MAX_BOUNDS,
            'CONDITION_DIM': CONDITION_DIM,
            'SPECTRA_LOSS_WEIGHT': SPECTRA_LOSS_WEIGHT,
            'METRIC_LOSS_WEIGHT': METRIC_LOSS_WEIGHT,
            'LAMBDA_PHYSICS': LAMBDA_PHYSICS,
            'EPOCHS': EPOCHS,
            'PRETRAIN_FWD_MODEL_EPOCHS': PRETRAIN_FWD_MODEL_EPOCHS,
        }
        
        # 执行验证和检测
        results = auto_detect_and_update_config(
            dataset_full_path, 
            current_config,
            save_path=os.path.join(project_root, 'config_analysis_results.json')
        )
        
        # 获取优化后的配置
        optimized = results['optimized_config']
        
        # 更新全局变量（在这里我们只更新关键参数）
        global STRUCT_MIN_BOUNDS, STRUCT_MAX_BOUNDS, BATCH_SIZE
        
        if AUTO_DETECT_BOUNDARIES:
            # 更新边界设置
            old_min_bounds = STRUCT_MIN_BOUNDS.copy()
            old_max_bounds = STRUCT_MAX_BOUNDS.copy()
            
            STRUCT_MIN_BOUNDS = optimized['STRUCT_MIN_BOUNDS']
            STRUCT_MAX_BOUNDS = optimized['STRUCT_MAX_BOUNDS']
            
            if STRUCT_MIN_BOUNDS != old_min_bounds or STRUCT_MAX_BOUNDS != old_max_bounds:
                print(f"ℹ️ 自动更新结构参数边界:")
                for i, param in enumerate(STRUCT_PARAMS):
                    if i < len(STRUCT_MIN_BOUNDS) and i < len(STRUCT_MAX_BOUNDS):
                        print(f"  {param}: [{old_min_bounds[i]:.4f}, {old_max_bounds[i]:.4f}] → "
                              f"[{STRUCT_MIN_BOUNDS[i]:.4f}, {STRUCT_MAX_BOUNDS[i]:.4f}]")
        
        # 打印验证结果
        if VALIDATE_CONFIG:
            validations = results['validations']
            has_warnings = False
            
            for validation_type, validation_result in validations.items():
                if validation_result.get('warnings'):
                    if not has_warnings:
                        print(f"⚠️ 配置验证警告:")
                        has_warnings = True
                    print(f"  {validation_type}:")
                    for warning in validation_result['warnings']:
                        print(f"    - {warning}")
                
                if validation_result.get('suggestions'):
                    print(f"Ὂ1 {validation_type} 建议:")
                    for suggestion in validation_result['suggestions']:
                        print(f"    - {suggestion}")
        
        print(f"✅ 配置验证完成，数据集大小: {results['total_samples']} 样本")
        
    except Exception as e:
        print(f"⚠️ 配置验证过程中出现错误: {e}")
        print("将使用默认配置继续")

# 在模块加载时自动执行配置验证
if __name__ != '__main__':  # 只在被导入时执行，避免直接运行时的问题
    validate_and_update_config()

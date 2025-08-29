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
AUTO_DETECT_BOUNDARIES = False  # 是否自动检测边界
VALIDATE_CONFIG = False         # 是否验证配置
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


STRUCT_MIN_BOUNDS = [0.0, 0.0, 0.0, 0.0] 
STRUCT_MAX_BOUNDS = [1.0, 1.0, 1.0, 1.0] 


ENFORCE_R1_GE_R2 = True

# --- Training Hyperparameters ---
# General
EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-4

# Forward Model Pre-training
PRETRAIN_FWD_MODEL_EPOCHS = 500
PRETRAIN_FWD_MODEL_LR = 5e-4
SPECTRA_LOSS_WEIGHT = 1.0  # Weight for spectra prediction loss
METRIC_LOSS_WEIGHT = 0.5   # Weight for metric prediction loss

# PI-GAN Training
GAN_EPOCHS = 1000
G_LR = 5e-5
D_LR = 5e-5
LATENT_DIM = 100


CONDITION_DIM = len(METRIC_PARAMS) # Assuming conditioning on all metrics

# Loss weights
LAMBDA_GP = 10  # Gradient Penalty
LAMBDA_R1 = 0.0  # R1 regularization weight (gamma/2). Set >0 to enable.
LAMBDA_PHYSICS = 0.2 # Physics-informed loss
LAMBDA_METRIC = 0.05 # Metric loss
LAMBDA_PID_FEEDBACK = 0.05 # Enable small weight for physical feedback initially
LAMBDA_TV = 0.001 # Total Variation loss for smoothness
PID_FEEDBACK_ANNEAL_EPOCHS = 200 # Linearly anneal PID weight from initial to target over these GAN epochs
PID_FEEDBACK_TARGET = 0.05 # Target PID weight at the end of annealing

# Diversity enhancement (Mode-Seeking)
LAMBDA_MODE_SEEKING = 0.1  # Weight for mode-seeking loss encouraging diverse outputs per condition
MODE_SEEKING_EPS = 1e-6    # Small epsilon to avoid division by zero
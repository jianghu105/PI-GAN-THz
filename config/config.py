
import torch

# --- General Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "dataset/THz_Metamaterial_Spectra_With_Metrics.csv"
CHECKPOINT_DIR = "checkpoints"
SAVED_MODELS_DIR = "saved_models"
LOG_DIR = "logs"
GENERATED_DATA_DIR = "generated_data"

# --- Data Preprocessing ---
STRUCT_PARAMS = ['r1', 'r2', 'w', 'g']
SPECTRA_PARAMS = [f'Freq_{i/100:.2f}' for i in range(50, 300)]
METRIC_PARAMS = ['f1', 'f2', 'Q1', 'FoM1', 'S1', 'Q2', 'FoM2', 'S2']
TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
RANDOM_STATE = 42

# --- Training Hyperparameters ---
# General
EPOCHS = 500
BATCH_SIZE = 64
LR = 1e-4

# Forward Model Pre-training
PRETRAIN_FWD_MODEL_EPOCHS = 300
PRETRAIN_FWD_MODEL_LR = 1e-3

# PI-GAN Training
GAN_EPOCHS = 1000
G_LR = 5e-5
D_LR = 5e-5
LATENT_DIM = 100

# Loss weights
LAMBDA_GP = 10  # Gradient Penalty
LAMBDA_PHYSICS = 0.5 # Physics-informed loss
LAMBDA_METRIC = 0.2 # Metric loss


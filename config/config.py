
import torch

# --- General Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "dataset/THz_Metamaterial_Spectra_With_Metrics.csv"
CHECKPOINT_DIR = "checkpoints"
SAVED_MODELS_DIR = "saved_models"
LOG_DIR = "logs"
GENERATED_DATA_DIR = "generated_data"
PLOT_DIR="plots"
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
SPECTRA_LOSS_WEIGHT = 0.5  # Weight for spectra prediction loss
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

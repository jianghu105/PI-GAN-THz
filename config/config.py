# PI_GAN_THZ/config/config.py

import os
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# --- 1. 项目根目录设置 ---
# 获取当前文件所在目录的父目录，作为项目的根目录
# 假设 config.py 在 PI_GAN_THZ/config/ 下
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 2. 通用设置 ---
RANDOM_SEED = 42                 # 随机种子，用于结果复现
# 自动检测设备 (CUDA 或 CPU)
DEVICE = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4                  # DataLoader 使用的进程数，可根据您的CPU核心数和内存调整


# --- 3. 路径设置 ---
# 数据文件目录和完整路径
DATA_DIR = os.path.join(PROJECT_ROOT, "dataset")
# 根据实际数据文件名调整
DATASET_PATH = os.path.join(DATA_DIR, "THz_Metamaterial_Spectra_With_Metrics.csv")  # 修改为实际数据文件名
FULL_DATA_PATH = os.path.join(DATA_DIR, "THz_Metamaterial_Spectra_With_Metrics.csv")  # 完整训练数据

# 模型检查点、最终模型和日志的保存目录
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints") # 训练过程中的检查点
SAVED_MODELS_DIR = os.path.join(PROJECT_ROOT, "saved_models") # 最终训练好的模型
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")                 # 日志文件目录
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")               # 绘图保存目录


# --- 4. 数据设置 ---
SPECTRUM_DIM = 250               # 太赫兹光谱的数据点数量 (例如，频率采样点)
NUM_SPECTRUM_POINTS = SPECTRUM_DIM  # 兼容性别名

# --- 5. 模型维度 ---
# 生成器 (Generator)
Z_DIM = 100                      # 噪声向量的维度，作为生成器的输入 (通常为100或128)
GENERATOR_INPUT_DIM = SPECTRUM_DIM               # 生成器输入维度（光谱）
GENERATOR_OUTPUT_DIM = 4                         # 生成器输出维度（结构参数）
GENERATOR_OUTPUT_PARAM_DIM = 4                   # 兼容性别名

# 判别器 (Discriminator)
DISCRIMINATOR_INPUT_SPEC_DIM = SPECTRUM_DIM      # D的输入光谱部分
DISCRIMINATOR_INPUT_PARAM_DIM = 4                # D的输入参数部分

# 前向模型 (ForwardModel)
FORWARD_MODEL_INPUT_DIM = 4                         # FwdModel的输入是结构参数
FORWARD_MODEL_OUTPUT_SPEC_DIM = SPECTRUM_DIM        # FwdModel的输出是光谱
FORWARD_MODEL_OUTPUT_METRICS_DIM = 8                # FwdModel的输出是物理指标 (f1, f2, Q1, FoM1, S1, Q2, FoM2, S2)


# --- 6. 训练设置 ---
# 前向模型预训练 (Pretraining Forward Model)
FWD_PRETRAIN_EPOCHS = 500  # 前向模型预训练的 epoch 数量
FWD_PRETRAIN_LR = 0.001    # 前向模型预训练的学习率
LR_FWD_SIM = 0.001         # 兼容性别名

# PI-GAN (Generator + Discriminator) 训练
NUM_EPOCHS = 500           # PI-GAN 主训练的 Epoch 数量
BATCH_SIZE = 64            # 训练批次大小

LR_G = 0.0002              # 生成器的学习率
LR_D = 0.0002              # 判别器的学习率

# 训练日志和模型保存频率
LOG_INTERVAL = 10          # 每隔多少个 batch 打印一次日志或 TensorBoard 记录
SAVE_MODEL_INTERVAL = 50   # 每隔多少个 epoch 保存一次模型检查点 (主要用于GAN模型)
SAVE_INTERVAL = 50         # 兼容性别名


# --- 7. 损失权重 (用于生成器总损失的加权求和) ---
# 这些权重需要根据实验结果和物理约束的重要性进行调整
# 请确保这些名称与您的 train_pigan.py 中实际使用的变量名匹配
LAMBDA_RECON = 100.0         # 光谱重建损失权重 (通常较大，因为是核心回归任务)
LAMBDA_PHYSICS = 10.0        # 物理指标重建损失权重 (如果细分为光谱和指标，此项可忽略)
LAMBDA_MAXWELL = 1.0         # 麦克斯韦方程组损失权重 (或光谱平滑性损失)
LAMBDA_LC = 1.0              # LC 模型近似损失权重
LAMBDA_PARAM_RANGE = 0.1     # 结构参数范围约束损失权重 (防止参数超出合理范围)
LAMBDA_BNN_KL = 0.0          # BNN KL 散度损失权重 (如果模型包含 Bayesian layers)

# 在 train_pigan.py 中更常见的物理一致性损失细分权重
LAMBDA_PHYSICS_SPECTRUM = 10.0 # 物理一致性损失 - 光谱部分
LAMBDA_PHYSICS_METRICS = 1.0   # 物理一致性损失 - 指标部分


# --- 8. 辅助函数 (可选，但推荐) ---
def create_directories():
    """在项目启动时创建所有必要的输出目录。"""
    os.makedirs(DATA_DIR, exist_ok=True) # 确保数据目录存在，尽管数据文件是用户提供的
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("所有必要的目录已确保存在。")

# 提示: 通常在主要的训练或运行脚本的 if __name__ == "__main__": 块中调用 create_directories()
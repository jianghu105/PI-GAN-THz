# PI_GAN_THZ/core/train/pretrain_fwd_model.py

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm.notebook import tqdm # <<<<<<< 更改：导入 tqdm.notebook 以兼容 Colab

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入 ForwardModel 和损失函数
from core.models.forward_model import ForwardModel
from core.utils.loss import criterion_mse # 用于光谱和指标的重建损失
import config.config as cfg # 导入 config
from core.utils.data_loader import MetamaterialDataset # 导入 MetamaterialDataset
from core.utils.set_seed import set_seed # 导入设置随机种子的函数

def pretrain_forward_model(forward_model: ForwardModel, dataloader: DataLoader, device: torch.device,
                           num_epochs: int, lr: float):
    """
    预训练前向仿真模型 (ForwardModel)。

    Args:
        forward_model (ForwardModel): 要预训练的 ForwardModel 实例。
        dataloader (DataLoader): 用于训练的数据加载器。
        device (torch.device): 训练设备 (CPU/GPU)。
        num_epochs (int): 预训练的 epoch 数量。
        lr (float): 预训练的学习率。
    Returns:
        list: 包含每个 epoch 平均损失的列表。
    """
    print("\n--- Pretraining Forward Model ---")

    # 定义优化器和损失函数
    optimizer = optim.Adam(forward_model.parameters(), lr=lr)
    mse_criterion = criterion_mse() # 实例化 MSE 损失

    # 初始化用于记录平均损失的列表
    epoch_losses = [] # 用于记录每个 epoch 的总平均损失

    # 设置模型为训练模式 (重要，因为 ForwardModel 包含 Dropout)
    forward_model.train()

    # 预训练循环
    for epoch in range(num_epochs):
        total_loss = 0.0
        # 使用 tqdm 包装 dataloader，显示进度条
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for i, (real_spectrum, _, real_params_norm, _, real_metrics_norm) in enumerate(progress_bar):
            # 将数据移动到指定设备
            real_spectrum = real_spectrum.to(device)
            real_params_norm = real_params_norm.to(device)
            real_metrics_norm = real_metrics_norm.to(device)

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播：预测光谱和指标
            predicted_spectrum, predicted_metrics_norm = forward_model(real_params_norm)

            # 计算损失：光谱重建损失 + 物理指标重建损失
            loss_spectrum = mse_criterion(predicted_spectrum, real_spectrum)
            loss_metrics = mse_criterion(predicted_metrics_norm, real_metrics_norm)

            # 总损失
            loss = loss_spectrum + loss_metrics

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 更新进度条的后缀信息
            progress_bar.set_postfix(
                Loss=f"{loss.item():.4f}", 
                SpecLoss=f"{loss_spectrum.item():.4f}", 
                MetricsLoss=f"{loss_metrics.item():.4f}"
            )

        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss) # 记录每个 epoch 的平均损失

        # 每个 epoch 结束时的日志记录
        print(f"Pretrain Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    # 预训练结束后保存模型
    os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_pretrained.pth")
    torch.save(forward_model.state_dict(), fwd_model_path)
    print(f"Pretrained ForwardModel saved to {fwd_model_path}")
    print("--- Forward Model Pretraining Complete ---")

    # 保存损失历史，以便后续评估脚本使用
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "fwd_pretrain_loss_history.pt")
    torch.save(epoch_losses, loss_history_path)
    print(f"Forward Model pretraining loss history saved to {loss_history_path}")

    return epoch_losses # 返回记录的损失列表

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain the Forward Model.")
    parser.add_argument('--epochs', type=int, default=cfg.FWD_PRETRAIN_EPOCHS,
                        help=f'Number of epochs for forward model pretraining (default: {cfg.FWD_PRETRAIN_EPOCHS})')
    parser.add_argument('--lr', type=float, default=cfg.FWD_PRETRAIN_LR,
                        help=f'Learning rate for forward model pretraining (default: {cfg.FWD_PRETRAIN_LR})')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE,
                        help=f'Batch size for pretraining (default: {cfg.BATCH_SIZE})')
    
    args = parser.parse_args()

    print("--- Starting Forward Model Pretraining Script ---")
    print(f"Arguments: {args}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 设置随机种子
    set_seed(cfg.RANDOM_SEED)
    print(f"Random seed set to: {cfg.RANDOM_SEED}")

    # --- 数据加载 ---
    # 确保创建必要的目录，包括 plots 目录
    cfg.create_directories() 

    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please check config.py and ensure the CSV file is there.")
        sys.exit(1) # 使用 sys.exit(1) 更明确地表示错误退出

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of batches: {len(dataloader)}")

    # --- 模型初始化 ---
    forward_model = ForwardModel(
        input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
        output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
        output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM
    ).to(device)

    print(f"ForwardModel Architecture:\n{forward_model}")

    # --- 调用预训练函数 ---
    # 这里不再直接进行评估和可视化，只执行训练并保存模型和损失历史
    pretrain_forward_model(
        forward_model=forward_model,
        dataloader=dataloader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr
    )

    print("--- Forward Model Pretraining Script Finished ---")
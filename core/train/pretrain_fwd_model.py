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
                           num_epochs: int, lr: float, log_interval: int = 10): # <<<<< 新增参数：log_interval
    """
    预训练前向仿真模型 (ForwardModel)。

    Args:
        forward_model (ForwardModel): 要预训练的 ForwardModel 实例。
        dataloader (DataLoader): 用于训练的数据加载器。
        device (torch.device): 训练设备 (CPU/GPU)。
        num_epochs (int): 预训练的 epoch 数量。
        lr (float): 预训练的学习率。
        log_interval (int): 每隔多少批次输出一次详细日志和更新进度条后缀。
    Returns:
        list: 包含每个 epoch 平均损失的列表。
    """
    tqdm.write("\n--- 正在预训练前向模型 ---") # <<<<< 使用 tqdm.write

    # 定义优化器和损失函数
    optimizer = optim.Adam(forward_model.parameters(), lr=lr)
    mse_criterion = criterion_mse() # 实例化 MSE 损失

    # 初始化用于记录平均损失的列表
    epoch_losses = [] # 用于记录每个 epoch 的总平均损失

    # 设置模型为训练模式 (重要，因为 ForwardModel 包含 Dropout)
    forward_model.train()

    # 预训练循环
    for epoch in range(num_epochs):
        # 初始化每个 epoch 的总损失
        epoch_total_loss = 0.0
        epoch_total_spectrum_loss = 0.0
        epoch_total_metrics_loss = 0.0
        
        # 初始化用于 log_interval 批次内的累积损失
        batch_accum_loss = 0.0
        batch_accum_spectrum_loss = 0.0
        batch_accum_metrics_loss = 0.0

        # 使用 tqdm 包装 dataloader，显示进度条
        # 尝试 leave=False，如果仍不显示，再改回 leave=True
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False) 
        
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

            # 累积损失以便计算 epoch 平均值
            epoch_total_loss += loss.item()
            epoch_total_spectrum_loss += loss_spectrum.item()
            epoch_total_metrics_loss += loss_metrics.item()

            # 累积用于 log_interval 批次内的损失
            batch_accum_loss += loss.item()
            batch_accum_spectrum_loss += loss_spectrum.item()
            batch_accum_metrics_loss += loss_metrics.item()

            # <<<<<<< 关键修改：每隔 log_interval 批次更新进度条后缀信息 >>>>>>>
            if (i + 1) % log_interval == 0:
                # 计算过去 log_interval 批次的平均损失
                current_avg_loss = batch_accum_loss / log_interval
                current_avg_spectrum_loss = batch_accum_spectrum_loss / log_interval
                current_avg_metrics_loss = batch_accum_metrics_loss / log_interval

                progress_bar.set_postfix(
                    Loss=f"{current_avg_loss:.4f}", 
                    SpecLoss=f"{current_avg_spectrum_loss:.4f}", 
                    MetricsLoss=f"{current_avg_metrics_loss:.4f}"
                )
                # 重置批次累积损失
                batch_accum_loss = 0.0
                batch_accum_spectrum_loss = 0.0
                batch_accum_metrics_loss = 0.0
        
        # 每个 epoch 结束时的最终平均损失
        avg_epoch_loss = epoch_total_loss / len(dataloader)
        avg_epoch_spectrum_loss = epoch_total_spectrum_loss / len(dataloader)
        avg_epoch_metrics_loss = epoch_total_metrics_loss / len(dataloader)
        epoch_losses.append(avg_epoch_loss) # 记录每个 epoch 的总平均损失

        # 使用 tqdm.write 确保在进度条下方打印信息，避免被覆盖
        tqdm.write(f"Epoch [{epoch+1}/{num_epochs}] - "
                   f"总平均损失: {avg_epoch_loss:.4f}, "
                   f"光谱损失: {avg_epoch_spectrum_loss:.4f}, "
                   f"指标损失: {avg_epoch_metrics_loss:.4f}")

    # 预训练结束后保存模型
    os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_pretrained.pth")
    torch.save(forward_model.state_dict(), fwd_model_path)
    tqdm.write(f"\n预训练的前向模型已保存到 {fwd_model_path}") # <<<<< 使用 tqdm.write
    tqdm.write("--- 前向模型预训练完成 ---") # <<<<< 使用 tqdm.write

    # 保存损失历史，以便后续评估脚本使用
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "fwd_pretrain_loss_history.pt")
    # 封装在字典中，以 'train_losses' 为键，兼容评估脚本
    torch.save({'train_losses': epoch_losses}, loss_history_path) 
    tqdm.write(f"前向模型预训练损失历史已保存到 {loss_history_path}") # <<<<< 使用 tqdm.write

    return epoch_losses # 返回记录的损失列表

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预训练前向模型。")
    parser.add_argument('--epochs', type=int, default=cfg.FWD_PRETRAIN_EPOCHS,
                        help=f'前向模型预训练的 epoch 数量 (默认: {cfg.FWD_PRETRAIN_EPOCHS})')
    parser.add_argument('--lr', type=float, default=cfg.FWD_PRETRAIN_LR,
                        help=f'前向模型预训练的学习率 (默认: {cfg.FWD_PRETRAIN_LR})')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE,
                        help=f'预训练的批次大小 (默认: {cfg.BATCH_SIZE})')
    parser.add_argument('--log_interval', type=int, default=10, # <<<<< 新增命令行参数
                        help='每隔多少批次输出一次详细日志和更新进度条后缀 (默认: 10)')
    
    args = parser.parse_args()

    tqdm.write("--- 正在启动前向模型预训练脚本 ---") # <<<<< 使用 tqdm.write
    tqdm.write(f"参数: {args}") # <<<<< 使用 tqdm.write

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tqdm.write(f"正在使用设备: {device}") # <<<<< 使用 tqdm.write
    set_seed(cfg.RANDOM_SEED)
    tqdm.write(f"随机种子已设置为: {cfg.RANDOM_SEED}") # <<<<< 使用 tqdm.write

    # --- 数据加载 ---
    # 确保创建必要的目录，包括 plots 目录
    cfg.create_directories() 

    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        tqdm.write(f"错误: 数据集未找到，路径为 {data_path}。请检查 config.py 并确保 CSV 文件存在。") # <<<<< 使用 tqdm.write
        sys.exit(1) # 使用 sys.exit(1) 更明确地表示错误退出

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True
    )
    tqdm.write(f"数据集大小: {len(dataset)} 个样本") # <<<<< 使用 tqdm.write
    tqdm.write(f"批次数量: {len(dataloader)}") # <<<<< 使用 tqdm.write

    # --- 模型初始化 ---
    forward_model = ForwardModel(
        input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
        output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
        output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM
    ).to(device)

    tqdm.write(f"ForwardModel 架构:\n{forward_model}") # <<<<< 使用 tqdm.write

    # --- 调用预训练函数 ---
    pretrain_forward_model(
        forward_model=forward_model,
        dataloader=dataloader,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        log_interval=args.log_interval # <<<<< 传入新的 log_interval 参数
    )

    tqdm.write("--- 前向模型预训练脚本已完成 ---") # <<<<< 使用 tqdm.write
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
import os
import sys
import argparse
import time
import sys
import numpy as np # 用于处理数据，例如 dataset.frequencies

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入所有模型
from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.models.forward_model import ForwardModel

# 导入所有需要的工具函数和损失函数
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_bce, criterion_mse, \
                             maxwell_equation_loss, lc_model_approx_loss, \
                             structural_param_range_loss, bnn_kl_loss

# 导入配置
import config.config as cfg


def train_pigan(dataloader: DataLoader, device: torch.device,
                generator: Generator, discriminator: Discriminator,
                forward_model: ForwardModel, dataset: MetamaterialDataset,
                num_epochs: int, log_interval: int = 10): # <<<<< 新增参数：log_interval
    """
    训练 PI-GAN 模型。

    Args:
        dataloader (DataLoader): 数据加载器。
        device (torch.device): 训练设备 (CPU/GPU)。
        generator (Generator): 生成器模型实例。
        discriminator (Discriminator): 判别器模型实例。
        forward_model (ForwardModel): 前向仿真模型实例 (已预训练)。
        dataset (MetamaterialDataset): 数据集实例，用于访问参数和指标的归一化范围。
        num_epochs (int): 训练的 epoch 数量。
        log_interval (int): 每隔多少批次输出一次详细日志和更新进度条后缀。
    Returns:
        dict: 包含所有损失历史的字典。
    """
    print("\n--- 正在启动 PI-GAN 训练 ---")

    # 定义优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.LR_G, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.LR_D, betas=(0.5, 0.999))
    
    # 添加学习率调度器
    # 生成器使用余弦退火调度，判别器使用步进调度
    scheduler_g = CosineAnnealingLR(optimizer_g, T_max=num_epochs, eta_min=cfg.LR_G * 0.01)
    scheduler_d = StepLR(optimizer_d, step_size=max(1, num_epochs // 4), gamma=0.5)
    
    print(f"学习率调度器初始化完成:")
    print(f"  Generator: CosineAnnealingLR (eta_min={cfg.LR_G * 0.01})")
    print(f"  Discriminator: StepLR (step_size={max(1, num_epochs // 4)}, gamma=0.5)")

    # 定义损失函数实例
    bce_criterion = criterion_bce()
    mse_criterion = criterion_mse()

    # 设置模型为训练/评估模式
    generator.train()
    discriminator.train()
    forward_model.eval() # ForwardModel 通常在 PI-GAN 训练中保持评估模式

    # 初始化用于记录的损失列表
    loss_history = {
        'd_losses': [],
        'g_losses': [],
        'adv_losses': [],
        'recon_spec_losses': [],
        'recon_metrics_losses': [],
        'maxwell_losses': [],
        'lc_losses': [],
        'param_range_losses': [],
        'bnn_kl_losses': []
    }

    # 训练循环
    for epoch in range(num_epochs):
        # 初始化每个 epoch 的总损失
        epoch_total_d_loss = 0.0
        epoch_total_g_loss = 0.0
        epoch_total_adv_loss = 0.0
        epoch_total_recon_loss_spec = 0.0
        epoch_total_recon_loss_metrics = 0.0
        epoch_total_maxwell_loss = 0.0
        epoch_total_lc_loss = 0.0
        epoch_total_param_range_loss = 0.0
        epoch_total_bnn_kl_loss = 0.0

        # 初始化用于 log_interval 批次内的累积损失
        batch_accum_d_loss = 0.0
        batch_accum_g_loss = 0.0
        batch_accum_adv_loss = 0.0

        # 自定义进度条实现，更适合Colab环境
        total_batches = len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Progress: [{'':50}] 0/{total_batches} batches")
        
        start_time = time.time()
        for i, (real_spectrum, real_params_denorm, real_params_norm, real_metrics_denorm, real_metrics_norm) in enumerate(dataloader):
            current_batch_size = real_spectrum.size(0)

            # Move data to the specified device
            real_spectrum = real_spectrum.to(device)
            real_params_denorm = real_params_denorm.to(device)
            real_params_norm = real_params_norm.to(device)
            real_metrics_norm = real_metrics_norm.to(device)

            # --- Train Discriminator (D) ---
            optimizer_d.zero_grad()

            # 标签平滑：使用0.9而不是1.0，0.1而不是0.0
            real_labels = torch.ones(current_batch_size, 1).to(device) * 0.9
            output_real = discriminator(real_spectrum, real_params_denorm)
            loss_d_real = bce_criterion(output_real, real_labels)

            predicted_params_norm = generator(real_spectrum)
            predicted_params_denorm_for_d = denormalize_params(predicted_params_norm.detach(), dataset.param_ranges)

            fake_labels = torch.zeros(current_batch_size, 1).to(device) + 0.1
            output_fake = discriminator(real_spectrum, predicted_params_denorm_for_d)
            loss_d_fake = bce_criterion(output_fake, fake_labels)

            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_d.step()

            # --- Train Generator (G) ---
            optimizer_g.zero_grad()

            predicted_params_norm = generator(real_spectrum)
            predicted_params_denorm_for_g = denormalize_params(predicted_params_norm, dataset.param_ranges)

            # 生成器训练时使用未平滑的标签(标准1.0)
            real_labels_for_g = torch.ones(current_batch_size, 1).to(device)
            output_g = discriminator(real_spectrum, predicted_params_denorm_for_g)
            loss_g_adv = bce_criterion(output_g, real_labels_for_g)

            with torch.no_grad():
                recon_spectrum, predicted_metrics_norm = forward_model(predicted_params_norm)

            loss_recon_spec = mse_criterion(recon_spectrum, real_spectrum)
            loss_recon_metrics = mse_criterion(predicted_metrics_norm, real_metrics_norm)

            frequencies_tensor = torch.tensor(dataset.frequencies, dtype=torch.float32, device=device).unsqueeze(0)
            loss_maxwell = maxwell_equation_loss(recon_spectrum, frequencies_tensor, predicted_params_norm)

            f1_idx = dataset.metric_name_to_idx['f1']
            f2_idx = dataset.metric_name_to_idx['f2']
            predicted_f1_norm = predicted_metrics_norm[:, f1_idx].unsqueeze(1)
            predicted_f2_norm = predicted_metrics_norm[:, f2_idx].unsqueeze(1)
            loss_lc = lc_model_approx_loss(predicted_f1_norm, predicted_f2_norm, predicted_params_norm)

            loss_param_range = structural_param_range_loss(predicted_params_norm)
            loss_bnn_kl = bnn_kl_loss(forward_model)

            loss_g_total = loss_g_adv + \
                           cfg.LAMBDA_RECON * loss_recon_spec + \
                           cfg.LAMBDA_PHYSICS_SPECTRUM * loss_recon_spec + \
                           cfg.LAMBDA_PHYSICS_METRICS * loss_recon_metrics + \
                           cfg.LAMBDA_MAXWELL * loss_maxwell + \
                           cfg.LAMBDA_LC * loss_lc + \
                           cfg.LAMBDA_PARAM_RANGE * loss_param_range + \
                           cfg.LAMBDA_BNN_KL * loss_bnn_kl

            loss_g_total.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            optimizer_g.step()

            # 累积当前批次的损失，用于计算 epoch 平均损失和 log_interval 期间的平均损失
            epoch_total_d_loss += loss_d.item()
            epoch_total_g_loss += loss_g_total.item()
            epoch_total_adv_loss += loss_g_adv.item()
            epoch_total_recon_loss_spec += loss_recon_spec.item()
            epoch_total_recon_loss_metrics += loss_recon_metrics.item()
            epoch_total_maxwell_loss += loss_maxwell.item()
            epoch_total_lc_loss += loss_lc.item()
            epoch_total_param_range_loss += loss_param_range.item()
            epoch_total_bnn_kl_loss += loss_bnn_kl.item()

            # 累积用于 log_interval 批次内的损失
            batch_accum_d_loss += loss_d.item()
            batch_accum_g_loss += loss_g_total.item()
            batch_accum_adv_loss += loss_g_adv.item()

            # 自定义进度条更新：每隔 log_interval 批次更新进度信息
            if (i + 1) % log_interval == 0:
                # 计算过去 log_interval 批次的平均损失
                current_avg_d_loss = batch_accum_d_loss / log_interval
                current_avg_g_loss = batch_accum_g_loss / log_interval
                current_avg_adv_loss = batch_accum_adv_loss / log_interval
                
                # 计算进度百分比和进度条
                progress = (i + 1) / total_batches
                filled_length = int(50 * progress)
                bar = '█' * filled_length + '-' * (50 - filled_length)
                
                # 计算剩余时间
                elapsed_time = time.time() - start_time
                if progress > 0:
                    eta = elapsed_time / progress * (1 - progress)
                    eta_str = f"ETA: {eta:.0f}s"
                else:
                    eta_str = "ETA: --"
                
                # 清除上一行并打印新的进度条
                print(f"\rProgress: [{bar}] {i+1}/{total_batches} | "
                      f"D:{current_avg_d_loss:.4f} G:{current_avg_g_loss:.4f} A:{current_avg_adv_loss:.4f} | "
                      f"{eta_str}", end='', flush=True)
                
                # 重置批次累积损失
                batch_accum_d_loss = 0.0
                batch_accum_g_loss = 0.0
                batch_accum_adv_loss = 0.0
        
        # Average losses for the epoch
        avg_d_loss = epoch_total_d_loss / len(dataloader)
        avg_g_loss = epoch_total_g_loss / len(dataloader)
        avg_adv_loss = epoch_total_adv_loss / len(dataloader)
        avg_recon_loss_spec = epoch_total_recon_loss_spec / len(dataloader)
        avg_recon_loss_metrics = epoch_total_recon_loss_metrics / len(dataloader)
        avg_maxwell_loss = epoch_total_maxwell_loss / len(dataloader)
        avg_lc_loss = epoch_total_lc_loss / len(dataloader)
        avg_param_range_loss = epoch_total_param_range_loss / len(dataloader)
        avg_bnn_kl_loss = epoch_total_bnn_kl_loss / len(dataloader)

        # 完成当前epoch后，打印最终进度条
        print(f"\rProgress: [{'█'*50}] {total_batches}/{total_batches} | "
              f"D:{avg_d_loss:.4f} G:{avg_g_loss:.4f} | "
              f"Completed in {time.time() - start_time:.0f}s")
        
        # 更新学习率调度器
        scheduler_g.step()
        scheduler_d.step()
        
        # 获取当前学习率
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        
        # 减少epoch输出频率，只在特定间隔输出详细信息
        if (epoch + 1) % cfg.LOG_INTERVAL == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}] Summary:")
            print(f"  D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}")
            print(f"  Learning Rates - G: {current_lr_g:.2e}, D: {current_lr_d:.2e}")
            print(f"  G_SubLosses - Adv: {avg_adv_loss:.4f}, "
                  f"Recon_Spec: {avg_recon_loss_spec:.4f}, "
                  f"Recon_Metrics: {avg_recon_loss_metrics:.4f}")
            print(f"  Physics_Losses - Maxwell: {avg_maxwell_loss:.4f}, "
                  f"LC: {avg_lc_loss:.4f}, ParamRange: {avg_param_range_loss:.4f}, "
                  f"BNN_KL: {avg_bnn_kl_loss:.4f}")
        
        # 每个epoch都记录损失到history中
        loss_history['d_losses'].append(avg_d_loss)
        loss_history['g_losses'].append(avg_g_loss)
        loss_history['adv_losses'].append(avg_adv_loss)
        loss_history['recon_spec_losses'].append(avg_recon_loss_spec)
        loss_history['recon_metrics_losses'].append(avg_recon_loss_metrics)
        loss_history['maxwell_losses'].append(avg_maxwell_loss)
        loss_history['lc_losses'].append(avg_lc_loss)
        loss_history['param_range_losses'].append(avg_param_range_loss)
        loss_history['bnn_kl_losses'].append(avg_bnn_kl_loss)


        # Save checkpoint
        if (epoch + 1) % cfg.SAVE_MODEL_INTERVAL == 0:
            os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, f"pigan_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'forward_model_state_dict': forward_model.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, checkpoint_path)
            print(f"已保存检查点到 {checkpoint_path}")

    print("--- PI-GAN 训练完成 ---")

    # Save final models after training
    os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth"))
    torch.save(forward_model.state_dict(), os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth"))
    print(f"最终模型已保存到 {cfg.SAVED_MODELS_DIR}")

    # 保存损失历史，以便后续评估脚本使用
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "pigan_loss_history.pt")
    torch.save(loss_history, loss_history_path)
    print(f"PI-GAN 训练损失历史已保存到 {loss_history_path}")

    return loss_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="直接运行 PI-GAN 训练。")
    parser.add_argument('--epochs', type=int, default=cfg.NUM_EPOCHS,
                        help=f'PI-GAN 训练的 epoch 数量 (默认: {cfg.NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE,
                        help=f'训练的批次大小 (默认: {cfg.BATCH_SIZE})')
    parser.add_argument('--lr_g', type=float, default=cfg.LR_G,
                        help=f'生成器的学习率 (默认: {cfg.LR_G})')
    parser.add_argument('--lr_d', type=float, default=cfg.LR_D,
                        help=f'判别器的学习率 (默认: {cfg.LR_D})')
    parser.add_argument('--fwd_model_path', type=str,
                        default=os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_pretrained.pth"),
                        help='预训练前向模型的路径。(默认: saved_models/forward_model_pretrained.pth)')
    parser.add_argument('--log_interval', type=int, default=10, # <<<<< 新增命令行参数
                        help='每隔多少批次输出一次详细日志和更新进度条后缀 (默认: 10)')
    
    args = parser.parse_args()

    print("--- 正在启动 PI-GAN 直接运行脚本 ---")
    print(f"参数: {args}")

    # 1. 设置设备和随机种子
    device = torch.device(cfg.DEVICE)
    print(f"正在使用设备: {device}")
    set_seed(cfg.RANDOM_SEED)
    print(f"随机种子已设置为: {cfg.RANDOM_SEED}")

    # 2. 创建必要的目录
    cfg.create_directories()

    # 3. 加载数据
    data_path = cfg.FULL_DATA_PATH
    if not os.path.exists(data_path):
        print(f"错误: 数据集未找到，路径为 {data_path}。请检查 config.py 并确保 CSV 文件存在。")
        sys.exit(1) # 如果数据集缺失则退出

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True # 使用 pin_memory 以加快数据传输到 GPU
    )
    print(f"数据集已加载，包含 {len(dataset)} 个样本。")
    print(f"每个 epoch 的批次数量: {len(dataloader)}")

    # 4. 初始化模型
    generator = Generator(input_dim=cfg.SPECTRUM_DIM, output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM).to(device)
    discriminator = Discriminator(input_spec_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM, 
                                  input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM).to(device)
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    print(f"生成器架构:\n{generator}")
    print(f"判别器架构:\n{discriminator}")
    print(f"前向模型架构:\n{forward_model}")

    # 5. 加载预训练的前向模型权重
    fwd_model_path = args.fwd_model_path
    if os.path.exists(fwd_model_path):
        print(f"正在从 {fwd_model_path} 加载预训练的前向模型...")
        forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device))
    else:
        print(f"错误: 预训练的前向模型未找到，路径为 {fwd_model_path}。")
        print("请确保已首先运行 'pretrain_fwd_model.py'，或指定正确的路径。")
        sys.exit(1) # 如果预训练模型缺失则退出

    # 6. 调用训练函数
    train_pigan(
        dataloader=dataloader,
        device=device,
        generator=generator,
        discriminator=discriminator,
        forward_model=forward_model,
        dataset=dataset,
        num_epochs=args.epochs,
        log_interval=args.log_interval # <<<<< 传入新的 log_interval 参数
    )

    print("\n--- PI-GAN 直接运行脚本已完成 ---")
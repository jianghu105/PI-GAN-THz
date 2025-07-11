# PI_GAN_THZ/core/train/train_pigan.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse
from tqdm.notebook import tqdm # <<<<<<< 更改：导入 tqdm.notebook 以兼容 Colab
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
                num_epochs: int):
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
    Returns:
        dict: 包含所有损失历史的字典。
    """
    print("\n--- Starting PI-GAN Training ---")

    # 定义优化器
    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.LR_G, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.LR_D, betas=(0.5, 0.999))

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
        total_d_loss = 0.0
        total_g_loss = 0.0
        total_adv_loss = 0.0
        total_recon_loss_spec = 0.0
        total_recon_loss_metrics = 0.0
        total_maxwell_loss = 0.0
        total_lc_loss = 0.0
        total_param_range_loss = 0.0
        total_bnn_kl_loss = 0.0

        # Wrap dataloader with tqdm for a progress bar
        data_iterator = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

        for i, (real_spectrum, real_params_denorm, real_params_norm, real_metrics_denorm, real_metrics_norm) in enumerate(data_iterator):
            current_batch_size = real_spectrum.size(0)

            # Move data to the specified device
            real_spectrum = real_spectrum.to(device)
            real_params_denorm = real_params_denorm.to(device)
            real_params_norm = real_params_norm.to(device)
            real_metrics_norm = real_metrics_norm.to(device)

            # --- Train Discriminator (D) ---
            optimizer_d.zero_grad()

            # 1. D's output on real data pairs (spectrum, real parameters)
            real_labels = torch.ones(current_batch_size, 1).to(device)
            output_real = discriminator(real_spectrum, real_params_denorm)
            loss_d_real = bce_criterion(output_real, real_labels)

            # 2. D's output on generated data pairs (spectrum, fake parameters)
            # Generate fake structural parameters (normalized)
            predicted_params_norm = generator(real_spectrum)
            # Denormalize fake parameters for Discriminator, then detach to stop gradients flowing back to G
            predicted_params_denorm_for_d = denormalize_params(predicted_params_norm.detach(), dataset.param_ranges)

            fake_labels = torch.zeros(current_batch_size, 1).to(device)
            output_fake = discriminator(real_spectrum, predicted_params_denorm_for_d)
            loss_d_fake = bce_criterion(output_fake, fake_labels)

            # Total D loss: real loss + fake loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_d.step()

            # --- Train Generator (G) ---
            optimizer_g.zero_grad()

            # G wants D to classify its fake data as real
            # Regenerate parameters to ensure gradient flow to G
            predicted_params_norm = generator(real_spectrum)
            predicted_params_denorm_for_g = denormalize_params(predicted_params_norm, dataset.param_ranges)

            output_g = discriminator(real_spectrum, predicted_params_denorm_for_g)
            loss_g_adv = bce_criterion(output_g, real_labels) # G aims for label 1 (real)

            # Physical information constraints and reconstruction losses
            # Pass generator's predicted parameters through the forward model
            with torch.no_grad(): # Ensure ForwardModel's weights are not updated during G's backward pass
                recon_spectrum, predicted_metrics_norm = forward_model(predicted_params_norm)

            # Spectrum reconstruction loss: predicted spectrum vs. real spectrum
            loss_recon_spec = mse_criterion(recon_spectrum, real_spectrum)

            # Physical metrics reconstruction loss: predicted metrics vs. real metrics
            loss_recon_metrics = mse_criterion(predicted_metrics_norm, real_metrics_norm)

            # Maxwell's equations loss
            # Ensure dataset.frequencies is available and converted to tensor correctly
            frequencies_tensor = torch.tensor(dataset.frequencies, dtype=torch.float32, device=device).unsqueeze(0)
            loss_maxwell = maxwell_equation_loss(recon_spectrum, frequencies_tensor, predicted_params_norm)

            # LC model approximation constraint loss
            f1_idx = dataset.metric_name_to_idx['f1']
            f2_idx = dataset.metric_name_to_idx['f2']
            predicted_f1_norm = predicted_metrics_norm[:, f1_idx].unsqueeze(1)
            predicted_f2_norm = predicted_metrics_norm[:, f2_idx].unsqueeze(1)
            loss_lc = lc_model_approx_loss(predicted_f1_norm, predicted_f2_norm, predicted_params_norm)

            # Structural parameter range loss
            loss_param_range = structural_param_range_loss(predicted_params_norm)

            # BNN KL divergence loss (0 if ForwardModel does not contain BNN layers)
            loss_bnn_kl = bnn_kl_loss(forward_model)

            # Total G loss = Adversarial Loss + Weighted Physical Constraint Losses
            loss_g_total = loss_g_adv + \
                           cfg.LAMBDA_RECON * loss_recon_spec + \
                           cfg.LAMBDA_PHYSICS_SPECTRUM * loss_recon_spec + \
                           cfg.LAMBDA_PHYSICS_METRICS * loss_recon_metrics + \
                           cfg.LAMBDA_MAXWELL * loss_maxwell + \
                           cfg.LAMBDA_LC * loss_lc + \
                           cfg.LAMBDA_PARAM_RANGE * loss_param_range + \
                           cfg.LAMBDA_BNN_KL * loss_bnn_kl

            loss_g_total.backward()
            optimizer_g.step()

            # Record losses for epoch average
            total_d_loss += loss_d.item()
            total_g_loss += loss_g_total.item()
            total_adv_loss += loss_g_adv.item()
            total_recon_loss_spec += loss_recon_spec.item()
            total_recon_loss_metrics += loss_recon_metrics.item()
            total_maxwell_loss += loss_maxwell.item()
            total_lc_loss += loss_lc.item()
            total_param_range_loss += loss_param_range.item()
            total_bnn_kl_loss += loss_bnn_kl.item()

            # Update tqdm postfix with current batch losses
            data_iterator.set_postfix(
                D_Loss=f"{loss_d.item():.4f}",
                G_Loss=f"{loss_g_total.item():.4f}",
                Adv=f"{loss_g_adv.item():.4f}"
            )

        # Average losses for the epoch
        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)
        avg_adv_loss = total_adv_loss / len(dataloader)
        avg_recon_loss_spec = total_recon_loss_spec / len(dataloader)
        avg_recon_loss_metrics = total_recon_loss_metrics / len(dataloader)
        avg_maxwell_loss = total_maxwell_loss / len(dataloader)
        avg_lc_loss = total_lc_loss / len(dataloader)
        avg_param_range_loss = total_param_range_loss / len(dataloader)
        avg_bnn_kl_loss = total_bnn_kl_loss / len(dataloader)

        # Print epoch summary
        if (epoch + 1) % cfg.LOG_INTERVAL == 0:
            print(f"\nEpoch [{epoch+1}/{num_epochs}]:")
            print(f"  D_Loss: {avg_d_loss:.4f}, G_Loss: {avg_g_loss:.4f}")
            print(f"  G_SubLosses - Adv: {avg_adv_loss:.4f}, "
                  f"Recon_Spec: {avg_recon_loss_spec:.4f}, "
                  f"Recon_Metrics: {avg_recon_loss_metrics:.4f}")
            print(f"  Physics_Losses - Maxwell: {avg_maxwell_loss:.4f}, "
                  f"LC: {avg_lc_loss:.4f}, ParamRange: {avg_param_range_loss:.4f}, "
                  f"BNN_KL: {avg_bnn_kl_loss:.4f}")
            
            # 记录平均损失到列表中
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
            print(f"Saved checkpoint to {checkpoint_path}")

    print("--- PI-GAN Training Complete ---")

    # Save final models after training
    os.makedirs(cfg.SAVED_MODELS_DIR, exist_ok=True)
    torch.save(generator.state_dict(), os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth"))
    torch.save(forward_model.state_dict(), os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth"))
    print(f"Final models saved to {cfg.SAVED_MODELS_DIR}")

    # 保存损失历史，以便后续评估脚本使用
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "pigan_loss_history.pt")
    torch.save(loss_history, loss_history_path)
    print(f"PI-GAN training loss history saved to {loss_history_path}")

    return loss_history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Directly run PI-GAN training.")
    parser.add_argument('--epochs', type=int, default=cfg.NUM_EPOCHS,
                        help=f'Number of epochs for PI-GAN training (default: {cfg.NUM_EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=cfg.BATCH_SIZE,
                        help=f'Batch size for training (default: {cfg.BATCH_SIZE})')
    parser.add_argument('--lr_g', type=float, default=cfg.LR_G,
                        help=f'Learning rate for Generator (default: {cfg.LR_G})')
    parser.add_argument('--lr_d', type=float, default=cfg.LR_D,
                        help=f'Learning rate for Discriminator (default: {cfg.LR_D})')
    parser.add_argument('--fwd_model_path', type=str,
                        default=os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_pretrained.pth"),
                        help='Path to the pretrained forward model. (Default: saved_models/forward_model_pretrained.pth)')
    
    args = parser.parse_args()

    print("--- Starting PI-GAN Direct Run ---")
    print(f"Arguments: {args}")

    # 1. Setup device and random seed
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")
    set_seed(cfg.RANDOM_SEED)
    print(f"Random seed set to: {cfg.RANDOM_SEED}")

    # 2. Create necessary directories
    cfg.create_directories()

    # 3. Load data
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}. Please check config.py and ensure the CSV file is there.")
        sys.exit(1) # Exit if dataset is missing

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True # Use pin_memory for faster data transfer to GPU
    )
    print(f"Dataset loaded with {len(dataset)} samples.")
    print(f"Number of batches per epoch: {len(dataloader)}")

    # 4. Initialize models
    generator = Generator(input_dim=cfg.SPECTRUM_DIM, output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM).to(device)
    discriminator = Discriminator(input_spec_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM, 
                                  input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM).to(device)
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    print(f"Generator Architecture:\n{generator}")
    print(f"Discriminator Architecture:\n{discriminator}")
    print(f"ForwardModel Architecture:\n{forward_model}")

    # 5. Load pretrained ForwardModel weights
    fwd_model_path = args.fwd_model_path
    if os.path.exists(fwd_model_path):
        print(f"Loading pretrained ForwardModel from: {fwd_model_path}")
        forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device))
    else:
        print(f"Error: Pretrained ForwardModel not found at {fwd_model_path}.")
        print("Please ensure the forward model has been pretrained by running 'pretrain_fwd_model.py' first, or specify the correct path.")
        sys.exit(1) # Exit if pretrained model is missing

    # 6. Call the training function
    # 这里不再直接进行评估和可视化，只执行训练并保存模型和损失历史
    train_pigan(
        dataloader=dataloader,
        device=device,
        generator=generator,
        discriminator=discriminator,
        forward_model=forward_model,
        dataset=dataset,
        num_epochs=args.epochs
    )

    print("\n--- PI-GAN Direct Run Complete ---")
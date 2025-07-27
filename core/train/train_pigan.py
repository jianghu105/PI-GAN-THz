import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import argparse
import time

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入所有模型
from core.models.generator import EnhancedGenerator
from core.models.discriminator import EnhancedDiscriminator
from core.models.forward_model import EnhancedForwardPINN

# 导入所有需要的工具函数和损失函数
from core.utils.data_loader import MetamaterialDataset
from core.utils.set_seed import set_seed
from core.utils.loss import generator_total_loss, criterion_bce
from core.utils.physics_constraints import find_peak_frequencies, extract_transmission

# 导入配置
import config.config as cfg

def verify_physical_feasibility(generator, forward_model, device, z_dim):
    """验证生成参数100%满足物理约束"""
    generator.eval()
    z = torch.randn(16, z_dim).to(device)
    # Note: The generator now only takes z as input
    params = generator(z)
    r1, r2, w, g = params.unbind(1)

    # 1. 检查r1 > r2
    r1_r2_ok = (r1 > r2).all().item()

    # 2. 检查参数范围
    r_min, r_max = 2.2e-6, 2.8e-6
    g_min, g_max = 1.8e-6, 3.0e-6
    w_min, w_max = 2.2e-6, 2.8e-6
    range_ok = (
        (r_min <= r1) & (r1 <= r_max) &
        (r_min <= r2) & (r2 <= r_max) &
        (w_min <= w) & (w <= w_max) &
        (g_min <= g) & (g <= g_max)
    ).all().item()

    # 3. 检查双峰关系
    spectra = forward_model(params)
    f1_peaks = find_peak_frequencies(spectra, 0.75e12, 0.95e12)
    f2_peaks = find_peak_frequencies(spectra, 1.95e12, 2.15e12)
    ratio_ok = ((f2_peaks / (f1_peaks + 1e-9)) > 2.2).all().item()

    # 4. 检查能量比
    s1_vals = extract_transmission(spectra, f1_peaks)
    s2_vals = extract_transmission(spectra, f2_peaks)
    energy_ratio = s1_vals / (s2_vals + 1e-9)
    energy_ok = ((energy_ratio < 0.1) & (energy_ratio > 0.05)).all().item()

    generator.train()
    return r1_r2_ok and range_ok and ratio_ok and energy_ok

def train_pigan(dataloader, device, generator, discriminator, forward_model, num_epochs, log_interval=10):
    print("\n--- Starting PI-GAN Training with Physics Constraints ---")

    optimizer_g = optim.Adam(generator.parameters(), lr=cfg.LR_G, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=cfg.LR_D, betas=(0.5, 0.999))
    bce_criterion = criterion_bce()

    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        forward_model.eval()

        # --- 三阶段训练策略 ---
        if epoch < 50:
            # 阶段1: 生成器主导训练 (冻结判别器)
            for p in discriminator.parameters(): p.requires_grad = False
            for p in generator.parameters(): p.requires_grad = True
        elif epoch < 100:
            # 阶段2: 判别器追赶 (冻结生成器)
            for p in discriminator.parameters(): p.requires_grad = True
            for p in generator.parameters(): p.requires_grad = False
        else:
            # 阶段3: 完整对抗训练
            for p in discriminator.parameters(): p.requires_grad = True
            for p in generator.parameters(): p.requires_grad = True

        for i, (real_spectrum, real_params, _, _, _) in enumerate(dataloader):
            real_spectrum = real_spectrum.to(device)
            real_params = real_params.to(device)
            batch_size = real_spectrum.size(0)
            z = torch.randn(batch_size, cfg.Z_DIM).to(device)

            # --- Train Discriminator ---
            if discriminator.training:
                optimizer_d.zero_grad()
                # Real samples
                real_labels = torch.ones(batch_size, 1).to(device) * 0.9
                d_real_output = discriminator(real_spectrum, real_params)
                loss_d_real = bce_criterion(d_real_output, real_labels)

                # Fake samples
                fake_params = generator(z)
                fake_spectrum = forward_model(fake_params.detach())
                fake_labels = torch.zeros(batch_size, 1).to(device) + 0.1
                d_fake_output = discriminator(fake_spectrum, fake_params.detach())
                loss_d_fake = bce_criterion(d_fake_output, fake_labels)
                
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()

            # --- Train Generator ---
            if generator.training:
                optimizer_g.zero_grad()
                loss_g, loss_components = generator_total_loss(
                    generator, discriminator, forward_model, z, real_spectrum, epoch, num_epochs
                )
                loss_g.backward()
                optimizer_g.step()

        # --- End of Epoch Logging ---
        print(f"Epoch [{epoch+1}/{num_epochs}] Completed.")
        if (epoch + 1) % 10 == 0:
            is_feasible = verify_physical_feasibility(generator, forward_model, device, cfg.Z_DIM)
            print(f"  Physical Feasibility Check: {'✅' if is_feasible else '❌'}")
            print(f"  Losses - G: {loss_g.item():.4f} | D: {loss_d.item():.4f}")
            print(f"    ├─ GAN: {loss_components['gan']:.4f}")
            print(f"    ├─ Spectral: {loss_components['spectral']:.4f} (λ={loss_components['lambda_data']:.2f})")
            print(f"    ├─ LC: {loss_components['lc']:.4f} (λ={loss_components['lambda_lc']:.2f})")
            print(f"    ├─ Maxwell: {loss_components['maxwell']:.4f} (λ={loss_components['lambda_maxwell']:.2f})")
            print(f"    └─ Energy: {loss_components['energy']:.4f} (λ={loss_components['lambda_energy']:.2f})")

    print("--- PI-GAN Training Finished ---")

if __name__ == '__main__':
    # 设置随机种子
    set_seed(cfg.RANDOM_SEED)
    
    # 准备设备
    device = torch.device(cfg.DEVICE)
    print(f"Using device: {device}")
    
    # 加载数据集
    print("Loading dataset...")
    dataset = MetamaterialDataset(cfg.FULL_DATA_PATH)
    
    # 划分训练集和验证集 (这里使用全部数据进行训练)
    train_loader = DataLoader(
        dataset, 
        batch_size=cfg.BATCH_SIZE, 
        shuffle=True, 
        num_workers=cfg.NUM_WORKERS
    )
    
    # 初始化模型
    print("Initializing models...")
    generator = EnhancedGenerator(
        spectrum_dim=cfg.SPECTRUM_DIM, 
        z_dim=cfg.Z_DIM, 
        output_dim=cfg.GENERATOR_OUTPUT_DIM
    ).to(device)
    
    discriminator = EnhancedDiscriminator(
        spectrum_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM, 
        param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM
    ).to(device)
    
    # 加载预训练的前向模型
    print("Loading pretrained forward model...")
    forward_model = EnhancedForwardPINN(
        input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM, 
        spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM
    ).to(device)
    
    # 尝试加载预训练权重
    try:
        forward_model.load_state_dict(torch.load(
            os.path.join(cfg.SAVED_MODELS_DIR, 'forward_model_enhanced_pretrained.pth'),
            map_location=device
        ))
        print("Successfully loaded pretrained forward model.")
    except:
        print("Warning: Failed to load pretrained forward model. Using random initialization.")
    
    # 运行训练
    print("Starting training...")
    train_pigan(
        train_loader, 
        device, 
        generator, 
        discriminator, 
        forward_model, 
        num_epochs=cfg.NUM_EPOCHS, 
        log_interval=cfg.LOG_INTERVAL
    )

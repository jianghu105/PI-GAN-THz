import sys
import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader 

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模型、数据加载器和绘图工具
from core.models.generator import Generator
from core.models.discriminator import Discriminator
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.plot_utils import plot_losses, plot_generated_samples
from core.utils.set_seed import set_seed

def evaluate_pigan(num_samples_to_plot: int = 5):
    """
    评估和可视化 PI-GAN 的训练结果。
    Args:
        num_samples_to_plot (int): 要可视化生成样本的图片数量。
    """
    print("\n--- 正在启动 PI-GAN 评估脚本 ---")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 设置随机种子，确保结果可复现
    set_seed(cfg.RANDOM_SEED)
    print(f"随机种子已设置为: {cfg.RANDOM_SEED}")

    # 确保创建必要的目录，包括 plots 目录
    cfg.create_directories() 
    print(f"评估图像将保存到: {cfg.PLOTS_DIR}")

    # --- 数据加载 ---
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        print(f"错误: 数据集未找到，路径为 {data_path}。请检查 config.py 并确保 CSV 文件存在。")
        # 在 Colab 中，sys.exit(1) 会立即终止单元格执行，可能看不到错误。
        # 为了调试，改为 return 并打印错误。
        return 
    
    # 注意：评估 PI-GAN 通常直接从数据集中取样，而不是通过 DataLoader 进行批处理循环。
    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    print(f"数据集大小: {len(dataset)} 个样本")

    # --- 模型初始化和加载 ---
    generator = Generator(input_dim=cfg.SPECTRUM_DIM, output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM).to(device)
    discriminator = Discriminator(input_spec_dim=cfg.DISCRIMINATOR_INPUT_SPEC_DIM, 
                                  input_param_dim=cfg.DISCRIMINATOR_INPUT_PARAM_DIM).to(device)
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    # 构造模型文件路径
    gen_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth")
    disc_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "discriminator_final.pth")
    # 假设前向模型在 PI-GAN 训练后也保存为 final 版本，如果不是，请使用 pretrain 版本
    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth") 

    # 检查所有模型文件是否存在
    models_found = True
    if not os.path.exists(gen_model_path):
        print(f"错误: Generator 模型未找到在 {gen_model_path}。")
        models_found = False
    if not os.path.exists(disc_model_path):
        print(f"错误: Discriminator 模型未找到在 {disc_model_path}。")
        models_found = False
    if not os.path.exists(fwd_model_path):
        print(f"错误: ForwardModel 模型未找到在 {fwd_model_path}。")
        models_found = False
    
    if not models_found:
        print("请确保 PI-GAN 训练已完成并保存了所有模型，或指定正确的路径。")
        return # 如果有模型缺失，优雅地返回

    try:
        generator.load_state_dict(torch.load(gen_model_path, map_location=device))
        discriminator.load_state_dict(torch.load(disc_model_path, map_location=device))
        forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device)) # 加载前向模型权重
    except Exception as e:
        print(f"错误: 加载模型时发生异常: {e}")
        print("请检查模型文件是否损坏或与模型架构不匹配。")
        return

    generator.eval() # 切换到评估模式
    discriminator.eval() # 切换到评估模式
    forward_model.eval() # 切换到评估模式
    print(f"已从 {cfg.SAVED_MODELS_DIR} 加载最终 Generator, Discriminator 和 ForwardModel。")

    # --- 加载损失历史 ---
    loss_history_path = os.path.join(cfg.SAVED_MODELS_DIR, "pigan_loss_history.pt")
    if not os.path.exists(loss_history_path):
        print(f"警告: PI-GAN 训练损失历史未找到，路径为 {loss_history_path}。无法生成损失图。")
        loss_history = {}
    else:
        try:
            # 假设 pigan_loss_history.pt 保存的是一个字典，包含各种损失列表
            loaded_history = torch.load(loss_history_path)
            # 检查关键键是否存在
            required_keys = ['g_losses', 'd_losses', 'adv_losses', 'recon_spec_losses', 
                             'recon_metrics_losses', 'maxwell_losses', 'lc_losses', 
                             'param_range_losses', 'bnn_kl_losses']
            
            if isinstance(loaded_history, dict) and all(k in loaded_history for k in required_keys):
                loss_history = loaded_history
                print(f"已从 {loss_history_path} 加载 PI-GAN 训练损失历史。")
            else:
                print(f"警告: 损失历史文件 {loss_history_path} 格式不正确或缺少关键数据。无法生成损失图。")
                loss_history = {}
        except Exception as e:
            print(f"错误: 加载损失历史时发生异常: {e}。无法生成损失图。")
            loss_history = {}

    # --- 绘制损失曲线 ---
    # 确保每个损失列表都不为空，且 epoch_for_plot 长度与损失列表匹配
    if loss_history and 'g_losses' in loss_history and loss_history['g_losses']:
        # 假设 loss_history 中的列表长度一致
        epochs_for_plot = list(range(1, len(loss_history['g_losses']) + 1)) 
        # 如果LOG_INTERVAL在训练时用于记录，你也可以根据需要调整epochs_for_plot
        # 例如：epochs_for_plot = [(i + 1) * cfg.LOG_INTERVAL for i in range(len(loss_history['g_losses']))]
        # 但通常对于损失曲线，直接使用 epoch 数更常见

        print("\n--- 正在生成 PI-GAN 训练损失图 ---")
        # 绘制总损失
        plot_losses(
            epochs=epochs_for_plot,
            losses={
                '生成器损失': loss_history.get('g_losses', []), # 使用 .get 防止键不存在
                '判别器损失': loss_history.get('d_losses', [])
            },
            title='生成器和判别器损失随 Epoch 变化',
            xlabel='Epoch',
            ylabel='损失',
            save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_gan_losses.png')
        )
        

        # 绘制生成器各子损失
        plot_losses(
            epochs=epochs_for_plot,
            losses={
                '对抗损失': loss_history.get('adv_losses', []),
                '光谱重建损失': loss_history.get('recon_spec_losses', []),
                '指标重建损失': loss_history.get('recon_metrics_losses', []),
                '麦克斯韦损失': loss_history.get('maxwell_losses', []),
                'LC 模型损失': loss_history.get('lc_losses', []),
                '参数范围损失': loss_history.get('param_range_losses', []),
                'BNN KL 散度损失': loss_history.get('bnn_kl_losses', [])
            },
            title='生成器子损失随 Epoch 变化',
            xlabel='Epoch',
            ylabel='损失',
            save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_generator_sub_losses.png')
        )
        
        print(f"损失图已保存到 {cfg.PLOTS_DIR}")
    else:
        print("没有可用于绘制的损失历史数据。")

    # --- 生成样本可视化 ---
    print("\n--- 正在生成 PI-GAN 样本可视化 ---")
    with torch.no_grad():
        if num_samples_to_plot <= 0:
            print("警告: 要绘制的样本数量为0或负数，跳过生成样本可视化。")
            
        elif num_samples_to_plot > len(dataset):
            print(f"警告: 要绘制的样本数量 ({num_samples_to_plot}) 超过数据集大小 ({len(dataset)})。将绘制所有 {len(dataset)} 个样本。")
            num_samples_to_plot = len(dataset)
        
        if num_samples_to_plot == 0: # 再次检查，如果调整后为0则退出
            print("没有足够的样本用于生成可视化。")
            
        else:
            # 随机选择一批真实光谱进行生成和可视化
            # ensure replace=False to avoid duplicate samples
            sample_indices = np.random.choice(len(dataset), num_samples_to_plot, replace=False) 
            
            # 使用 DataLoader 来高效地获取这些样本，即使是单个批次也推荐
            from torch.utils.data import Subset
            subset_dataset = Subset(dataset, sample_indices)
            sample_dataloader = DataLoader(subset_dataset, batch_size=num_samples_to_plot, shuffle=False)
            
            # 从 dataloader 中获取第一个（也是唯一的）批次
            # real_spectrum, real_params_denorm, real_params_norm, real_metrics_denorm, real_metrics_norm
            sample_batch = next(iter(sample_dataloader)) 
            
            sample_real_spectrums = sample_batch[0].to(device)
            sample_real_params_denorm = sample_batch[1].to(device) # 真实未归一化参数

            # 通过生成器预测参数
            sample_predicted_params_norm = generator(sample_real_spectrums)
            sample_predicted_params_denorm = denormalize_params(sample_predicted_params_norm, dataset.param_ranges)

            # 通过前向模型重构光谱 (用于循环一致性检查)
            recon_spectrums, _ = forward_model(sample_predicted_params_norm)

            # 将张量移动回CPU并转换为NumPy以便绘图
            sample_real_spectrums_np = sample_real_spectrums.cpu().numpy()
            recon_spectrums_np = recon_spectrums.cpu().numpy()
            sample_real_params_denorm_np = sample_real_params_denorm.cpu().numpy()
            sample_predicted_params_denorm_np = sample_predicted_params_denorm.cpu().numpy()
            
            frequencies = dataset.frequencies # 从 MetamaterialDataset 中获取频率

            plot_generated_samples(
                real_spectrums=sample_real_spectrums_np,
                recon_spectrums=recon_spectrums_np,
                real_params=sample_real_params_denorm_np,
                predicted_params=sample_predicted_params_denorm_np,
                frequencies=frequencies,
                num_samples=num_samples_to_plot,
                save_path=os.path.join(cfg.PLOTS_DIR, 'pigan_generated_samples.png')
            )
            
            print(f"PI-GAN 样本图已保存到 {cfg.PLOTS_DIR}")
    
    print("--- PI-GAN 评估脚本已完成 ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估已训练的 PI-GAN 模型。")
    parser.add_argument('--num_samples', type=int, default=5,
                        help='要可视化绘制的样本数量 (默认: 5)')
    args = parser.parse_args()
    
    evaluate_pigan(num_samples_to_plot=args.num_samples)
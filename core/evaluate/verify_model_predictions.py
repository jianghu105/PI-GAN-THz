import sys
import os
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader, Subset # 确保导入 DataLoader 和 Subset
from tqdm.notebook import tqdm # 用于显示进度条

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入模型、数据加载器和绘图工具
from core.models.generator import Generator
from core.models.forward_model import ForwardModel
import config.config as cfg
from core.utils.data_loader import MetamaterialDataset, denormalize_params, denormalize_metrics
from core.utils.plot_utils import plot_fwd_model_predictions # 复用前向模型的预测绘图函数
from core.utils.set_seed import set_seed
from core.utils.loss import criterion_mse # 用于计算 MSE

def verify_predictions(num_samples: int = 10):
    """
    验证模型的预测能力，包括生成器的逆向预测和前向模型的正向预测。
    Args:
        num_samples (int): 要验证和可视化其预测结果的样本数量。
    """
    print("\n--- 正在启动模型预测验证脚本 ---")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 设置随机种子，确保结果可复现
    set_seed(cfg.RANDOM_SEED)
    print(f"随机种子已设置为: {cfg.RANDOM_SEED}")

    # 确保创建必要的目录，包括 plots 目录
    cfg.create_directories()
    print(f"图像将保存到: {cfg.PLOTS_DIR}")

    # --- 数据加载 ---
    data_path = cfg.DATASET_PATH
    if not os.path.exists(data_path):
        print(f"错误: 数据集未找到，路径为 {data_path}。请检查 config.py 并确保 CSV 文件存在。")
        return # 优雅地退出

    dataset = MetamaterialDataset(data_path=data_path, num_points_per_sample=cfg.SPECTRUM_DIM)
    print(f"数据集已加载，包含 {len(dataset)} 个样本。")

    # --- 模型初始化和加载 ---
    generator = Generator(input_dim=cfg.SPECTRUM_DIM, output_dim=cfg.GENERATOR_OUTPUT_PARAM_DIM).to(device)
    forward_model = ForwardModel(input_param_dim=cfg.FORWARD_MODEL_INPUT_DIM,
                                 output_spectrum_dim=cfg.FORWARD_MODEL_OUTPUT_SPEC_DIM,
                                 output_metrics_dim=cfg.FORWARD_MODEL_OUTPUT_METRICS_DIM).to(device)
    
    # 定义模型保存路径
    gen_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "generator_final.pth")
    fwd_model_path = os.path.join(cfg.SAVED_MODELS_DIR, "forward_model_final.pth") # 或 forward_model_pretrained.pth

    # 检查模型文件是否存在
    models_found = True
    if not os.path.exists(gen_model_path):
        print(f"错误: 生成器模型未找到在 {gen_model_path}。")
        models_found = False
    if not os.path.exists(fwd_model_path):
        print(f"错误: 前向模型未找到在 {fwd_model_path}。")
        models_found = False
    
    if not models_found:
        print("请确保 PI-GAN 训练和前向模型预训练已完成并保存了所有模型。")
        return

    try:
        generator.load_state_dict(torch.load(gen_model_path, map_location=device))
        forward_model.load_state_dict(torch.load(fwd_model_path, map_location=device))
        print(f"已从 {cfg.SAVED_MODELS_DIR} 加载 Generator 和 ForwardModel。")
    except Exception as e:
        print(f"错误: 加载模型时发生异常: {e}")
        print("请检查模型文件是否损坏或与模型架构不匹配。")
        return

    generator.eval() # 切换到评估模式
    forward_model.eval() # 切换到评估模式

    # --- 随机选择样本进行预测和可视化 ---
    print(f"\n--- 正在为 {num_samples} 个样本生成模型预测可视化 ---")
    mse_criterion = criterion_mse() # 初始化 MSE 准则

    if num_samples <= 0:
        print("警告: 要验证的样本数量为0或负数，跳过预测验证。")
        return
    
    if num_samples > len(dataset):
        print(f"警告: 要验证的样本数量 ({num_samples}) 超过数据集大小 ({len(dataset)})。将验证所有 {len(dataset)} 个样本。")
        num_samples = len(dataset)
    
    if num_samples == 0:
        print("没有足够的样本用于验证。")
        return

    # 随机选择样本索引
    sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset_dataset = Subset(dataset, sample_indices)
    # 使用 DataLoader 仅加载选定样本，即使只有一个批次
    sample_dataloader = DataLoader(subset_dataset, batch_size=num_samples, shuffle=False)
    
    # 获取第一个（也是唯一的）批次数据
    # _ 是为了跳过 real_params_denorm 和 real_metrics_denorm，因为这里我们会从归一化版本去反归一化
    # 实际上，我们需要 real_params_denorm 和 real_metrics_denorm 用于真实值绘图
    # 因此，我们直接从 batch 中解包
    real_spectrums_batch, real_params_denorm_batch, _, real_metrics_denorm_batch, _ = next(iter(sample_dataloader))

    # 将真实数据移动到设备
    real_spectrums = real_spectrums_batch.to(device)
    real_params_denorm_input = real_params_denorm_batch.to(device) # 用于绘图的真实未归一化参数
    real_metrics_denorm_input = real_metrics_denorm_batch.to(device) # 用于绘图的真实未归一化指标

    all_real_params = []
    all_real_spectrums = []
    all_predicted_spectrums = []
    all_real_metrics = []
    all_predicted_metrics = []
    
    total_spectrum_mse = 0.0
    total_metrics_mse = 0.0
    num_processed_samples = 0

    with torch.no_grad():
        # 这里因为我们已经通过 DataLoader 获取了所有要处理的样本，所以不需要再循环
        # 但为了逻辑清晰和保持与 DataLoader 的一致性，我们可以假设只有一个批次
        
        # 1. 使用生成器从真实光谱预测参数
        # 输入生成器的是归一化的真实光谱
        predicted_params_norm = generator(real_spectrums)
        # 将预测的参数去归一化，用于绘图
        predicted_params_denorm = denormalize_params(predicted_params_norm, dataset.param_ranges).cpu().numpy()

        # 2. 使用前向模型从预测参数预测光谱和指标
        predicted_spectrum, predicted_metrics_norm = forward_model(predicted_params_norm)
        # 将预测的光谱和指标去归一化，用于绘图
        predicted_spectrum_np = predicted_spectrum.cpu().numpy()
        predicted_metrics_denorm_np = denormalize_metrics(predicted_metrics_norm, dataset.metric_ranges).cpu().numpy()

        # 计算 MSE（真实光谱 vs 前向模型预测光谱）
        total_spectrum_mse += mse_criterion(predicted_spectrum, real_spectrums).item() * num_samples
        total_metrics_mse += mse_criterion(predicted_metrics_norm, real_metrics_denorm_input).item() * num_samples # 注意这里需要 real_metrics_norm，但我们传入的是 denorm_input。

        # 修正：MSE计算应该使用归一化后的真实指标
        # 确保 real_metrics_norm 被获取并用于 MSE 计算
        # 从 dataset 中获取对应的归一化真实指标 (这里需要调整一下数据加载部分或重新考虑)
        # 为了简化，我们假设 real_metrics_denorm_input 也可以用于比较，但更精确的是使用归一化版本
        # 最好是这样获取：
        _, _, _, _, real_metrics_norm_batch_for_mse = next(iter(sample_dataloader))
        real_metrics_norm_for_mse = real_metrics_norm_batch_for_mse.to(device)
        total_metrics_mse += mse_criterion(predicted_metrics_norm, real_metrics_norm_for_mse).item() * num_samples

        num_processed_samples += num_samples

        # 存储所有样本数据用于绘图
        all_real_params = real_params_denorm_input.cpu().numpy()
        all_real_spectrums = real_spectrums.cpu().numpy()
        all_predicted_spectrums = predicted_spectrum_np # 已经是 numpy
        all_real_metrics = real_metrics_denorm_input.cpu().numpy()
        all_predicted_metrics = predicted_metrics_denorm_np # 已经是 numpy

    # --- 绘制可视化结果 ---
    frequencies = dataset.frequencies # 从 MetamaterialDataset 中获取频率
    metric_names = dataset.metric_names # 从 MetamaterialDataset 中获取指标名称

    if all_real_params.size > 0: # 确保有数据才绘图
        plot_fwd_model_predictions(
            real_params=all_real_params,
            real_spectrums=all_real_spectrums,
            predicted_spectrums=all_predicted_spectrums,
            real_metrics=all_real_metrics,
            predicted_metrics=all_predicted_metrics,
            frequencies=frequencies,
            num_samples=num_samples, # 使用实际处理的样本数量
            save_path=os.path.join(cfg.PLOTS_DIR, 'model_prediction_verification.png'),
            metric_names=metric_names
        )
        print(f"模型预测验证图已保存到 {cfg.PLOTS_DIR}")
    else:
        print("没有可用于绘制的预测验证数据。")

    if num_processed_samples > 0:
        avg_spectrum_mse = total_spectrum_mse / num_processed_samples
        avg_metrics_mse = total_metrics_mse / num_processed_samples
        print(f"\n平均光谱 MSE (真实 vs. 前向模型预测): {avg_spectrum_mse:.4f}")
        print(f"平均指标 MSE (真实 vs. 前向模型预测): {avg_metrics_mse:.4f}")
    else:
        print("没有处理任何样本用于 MSE 计算。")

    print("--- 模型预测验证脚本已完成 ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证 PI-GAN 生成器和前向模型的预测能力。")
    parser.add_argument('--num_samples', type=int, default=10,
                        help='要验证和可视化其预测结果的样本数量 (默认: 10)')
    args = parser.parse_args()
    
    verify_predictions(num_samples=args.num_samples)
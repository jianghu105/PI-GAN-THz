# PI_GAN_THZ/core/utils/set_seed.py

import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """
    设置所有随机性来源的种子，以确保实验的可重复性。

    Args:
        seed (int): 用于设置随机种子的整数值。
    """
    # Python 的 random 模块
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    
    # 如果使用 CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 为所有 GPU 设置种子
        # 确保在使用 cuDNN 时结果是确定的
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # 关闭 cuDNN 自动寻找最佳算法，以确保确定性
    
    # 设置环境变量，这对一些外部库可能有帮助 (例如，某些 CUDNN 操作)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 对于 PyTorch DataLoader 的 worker，可能需要额外的种子设置，
    # 但对于大多数情况，上述设置已经足够。
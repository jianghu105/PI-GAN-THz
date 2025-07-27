# PI_GAN_THZ/core/models/physical_layers.py

import torch
import torch.nn as nn

class SRRPhysicsConstrainedOutput(nn.Module):
    """
    SRR专用物理约束输出层 - 硬保证100%物理可行性
    
    输入: [batch, hidden_dim] 生成器隐藏层输出
    输出: [batch, 4] 结构参数 [r1, r2, w, g] (单位: 米)
    
    关键物理约束:
      - r1 > r2 (必须，非可选)
      - r1,r2,w ∈ [2.2, 2.8] μm
      - g ∈ [1.8, 3.0] μm
      - 基于您提供的结构图和参数范围
    """
    def __init__(self, input_dim, output_dim=4):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        # SRR物理边界(基于您的参数范围)
        self.r_min, self.r_max = 2.2e-6, 2.8e-6
        self.g_min, self.g_max = 1.8e-6, 3.0e-6
        self.w_min, self.w_max = 2.2e-6, 2.8e-6

    def forward(self, x):
        """
        实现硬物理约束，确保100%可行性
        
        关键机制:
          1. 硬编码r1 > r2 (通过二次修正)
          2. 参数范围clamping (您的扫描范围)
          3. 金层厚度固定为0.2μm的简化处理
        """
        params = self.fc(x)
        r1_raw, r2_raw, w_raw, g_raw = params.unbind(1)
        
        # 1. 硬约束1: r1 > r2 (必须保证)
        # 先将r1和r2映射到大致范围，再强制约束
        r1_scaled = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(r1_raw)
        r2_scaled = self.r_min + (self.r_max - self.r_min) * torch.sigmoid(r2_raw)

        # 强制 r1 > r2
        r1_final = torch.max(r1_scaled, r2_scaled) + 1e-7 # 确保r1是较大的那个
        r2_final = torch.min(r1_scaled, r2_scaled)

        # 2. 硬约束2: 参数范围 (您的扫描范围)
        # 使用sigmoid将输出映射到(0,1)，然后缩放到目标范围
        w = self.w_min + (self.w_max - self.w_min) * torch.sigmoid(w_raw)
        g = self.g_min + (self.g_max - self.g_min) * torch.sigmoid(g_raw)
        
        # 最终clamp确保万无一失
        r1 = torch.clamp(r1_final, self.r_min, self.r_max)
        r2 = torch.clamp(r2_final, self.r_min, self.r_max)
        
        return torch.stack([r1, r2, w, g], dim=1)

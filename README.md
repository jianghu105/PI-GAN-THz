# PI-GAN-THz

---

## 文件结构说明

```
PI_GAN_THz/
├─ core/
│   ├─ models/         # 神经网络模型（生成器、判别器、前向模型等）
│   ├─ train/          # 训练脚本
│   ├─ eval/           # 评估与测试脚本
│   ├─ utils/          # 工具函数（数据加载、日志等）
│   └─ visual/         # 可视化相关代码
├─ config/             # 配置文件（如config.yaml）
├─ dataset/            # 数据集及说明
├─ checkpoints/        # 训练中间权重
├─ saved_models/       # 最优模型保存
├─ colab/              # Colab平台notebook
├─ notebooks/          # 研究/实验notebook
├─ requirements.txt    # 依赖包
└─ README.md           # 项目说明
```

## 环境依赖与安装

建议使用Python 3.8+，主要依赖如下：

```
pandas
scikit-learn
torch
tqdm
matplotlib
seaborn
```

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备说明

- 数据格式为CSV，每行包含结构参数（r1, r2, w, g）、250维频谱、Q值、FoM等
- 数据样例：
  | r1 | r2 | w | g | s0 | s1 | ... | s249 | Q | FoM | peak_shift |
  |----|----|---|---|----|----|-----|------|---|-----|-----------|

- 数据加载与归一化见 `core/utils/data_loader.py`

## 快速开始

### 标准模型训练
1. **数据加载与预处理**
   - 修改 `config/config.yaml` 或脚本参数，指定数据路径
   - 使用 `core/utils/data_loader.py` 加载和划分数据

2. **模型训练**
   - 运行 `core/train/train_pigan.py` 进行GAN+PINN训练
   - 训练日志和模型权重自动保存

3. **模型评估与推理**
   - 使用 `core/eval/` 下脚本进行模型评估
   - 可视化结果见 `core/visual/`

### 增强模型训练
1. **使用增强模型训练脚本**
   - 运行 `train_enhanced.sh` 脚本进行端到端训练和评估：
   ```bash
   ./train_enhanced.sh
   ```

2. **分别训练各组件**
   - 预训练增强版前向模型:
   ```bash
   python core/train/pretrain_fwd_model_enhanced.py --epochs 300
   ```
   
   - 训练增强版PI-GAN:
   ```bash
   python core/train/train_pigan_enhanced.py --epochs 300
   ```

3. **评估增强模型**
   ```bash
   python core/evaluate/unified_evaluator.py --enhanced
   ```

## 模型架构说明

### 标准模型
- **生成器**: 简单的全连接网络，将光谱映射到结构参数
- **判别器**: 全连接网络，判断光谱-参数对的真实性
- **前向模型**: 全连接网络，将结构参数映射到光谱和物理指标

### 增强模型
- **增强生成器**: 
  - 光谱特征提取分支（1D CNN）
  - 潜在向量处理分支
  - 物理约束输出层（带硬编码边界约束）

- **增强判别器**:
  - 共享的光谱处理分支（共振区域聚焦）
  - 结构参数特征提取
  - 双判别头设计（真实/生成判别 + 物理一致性判别）

- **增强前向PINN**:
  - 参数特征提取模块（多层全连接 + SiLU激活）
  - 傅里叶特征编码器
  - 频谱生成核心（带残差块）
  - 区域化物理约束计算

## Colab平台体验

- 参考 `colab/` 下notebook，快速体验训练与推理流程
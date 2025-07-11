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

1. **数据加载与预处理**
   - 修改 `config/config.yaml` 或脚本参数，指定数据路径
   - 使用 `core/utils/data_loader.py` 加载和划分数据

2. **模型训练**
   - 运行 `core/train/train_pigan.py` 进行GAN+PINN训练
   - 训练日志和模型权重自动保存

3. **模型评估与推理**
   - 使用 `core/eval/` 下脚本进行模型评估
   - 可视化结果见 `core/visual/`

4. **Colab平台体验**
   - 参考 `colab/` 下notebook，快速体验训练与推理流程

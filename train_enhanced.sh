#!/bin/bash
# PI_GAN_THz/train_enhanced.sh

echo "=================================="
echo "Training Enhanced PI-GAN for THz Metamaterials"
echo "=================================="

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:/mnt/d/PI_GAN_THz"

# 1. 预训练增强版前向模型
echo "Step 1: Pretraining Enhanced Forward Model..."
python /mnt/d/PI_GAN_THz/core/train/pretrain_fwd_model_enhanced.py \
  --epochs 300 \
  --lr 0.001 \
  --batch_size 64 \
  --log_interval 5

# 检查前向模型是否训练成功
if [ $? -ne 0 ]; then
  echo "Error: Failed to pretrain enhanced forward model"
  exit 1
fi

echo "Enhanced forward model pretraining completed successfully!"

# 2. 训练增强版PI-GAN
echo "Step 2: Training Enhanced PI-GAN..."
python /mnt/d/PI_GAN_THz/core/train/train_pigan_enhanced.py \
  --epochs 300 \
  --batch_size 64 \
  --lr_g 0.0002 \
  --lr_d 0.0002 \
  --log_interval 5

# 检查PI-GAN是否训练成功
if [ $? -ne 0 ]; then
  echo "Error: Failed to train enhanced PI-GAN"
  exit 1
fi

echo "Enhanced PI-GAN training completed successfully!"

# 3. 评估增强版模型
echo "Step 3: Evaluating Enhanced Models..."
python /mnt/d/PI_GAN_THz/core/evaluate/unified_evaluator.py \
  --num_samples 500

# 检查评估是否成功
if [ $? -ne 0 ]; then
  echo "Error: Failed to evaluate enhanced models"
  exit 1
fi

echo "Enhanced model evaluation completed successfully!"

echo "=================================="
echo "Enhanced PI-GAN Training Pipeline Completed!"
echo "=================================="
#!/bin/bash

# Exchange Rate Experiment with MPS Optimization for macOS
# Optimized for Apple Silicon Macs

echo "=========================================="
echo "Exchange Rate Dataset - MPS Accelerated"
echo "=========================================="

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 创建日志目录
mkdir -p ./logs/ema

# 提示用户激活环境
echo "注意: 请确保已经激活xpatch环境:"
echo "conda activate xpatch"
echo ""

# 设置MPS环境变量（可选，用于调试）
export PYTORCH_ENABLE_MPS_FALLBACK=1

# MPS优化的参数设置
ma_type=ema
alpha=0.3
beta=0.3
model_name=xPatch
seq_len=96

echo "设备信息检查..."
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    print('将使用MPS加速')
else:
    print('MPS不可用，将使用CPU')
"

for pred_len in 96 192 336 720
do
  echo "运行实验: pred_len=$pred_len (使用MPS加速)"
  
  # 参数说明：
  # --is_training 1: 训练模式 (1=训练, 0=测试)
  # --root_path: 数据集根目录路径
  # --data_path: 具体数据文件路径 (汇率数据)
  # --model_id: 动态模型ID (包含预测长度和移动平均类型)
  # --model: 使用的模型名称 (xPatch)
  # --data: 数据加载器类型 (custom表示自定义数据集)
  # --features: 特征类型 (M=多变量, S=单变量, MS=多变量预测单变量)
  # --seq_len: 输入序列长度 (96个历史时间步)
  # --pred_len: 预测序列长度 (动态：96/192/336/720)
  # --enc_in: 编码器输入特征维度 (8种汇率)
  # --des: 实验描述标识符
  # --itr: 实验重复次数 (取平均值)
  # --batch_size: 批大小 (32适合MPS加速和macOS内存)
  # --learning_rate: 学习率 (Adam优化器)
  # --lradj: 学习率调度策略 (sigmoid衰减)
  # --ma_type: 移动平均类型 (ema=指数移动平均)
  # --alpha: EMA的alpha参数 (0.3平滑系数)
  # --beta: DEMA的beta参数 (备用)
  # --num_workers: 数据加载进程数 (macOS设为0避免multiprocessing问题)
  # --use_gpu: 启用GPU加速 (自动选择MPS优先)
  # --train_epochs: 训练轮数 (完整训练10轮)
  # --patience: 早停耐心值，输出重定向到日志文件
  
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate/exchange_rate.csv \
    --model_id exchange_${pred_len}_${ma_type}_mps \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --enc_in 8 \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.00001 \
    --lradj 'sigmoid' \
    --ma_type $ma_type \
    --alpha $alpha \
    --beta $beta \
    --num_workers 0 \
    --use_gpu True \
    --train_epochs 10 \
    --patience 3 > logs/$ma_type/${model_name}_exchange_${seq_len}_${pred_len}_mps.log
    
  echo "完成: pred_len=$pred_len"
  echo "日志保存至: logs/$ma_type/${model_name}_exchange_${seq_len}_${pred_len}_mps.log"
  echo "------------------------------------------"
done

echo "=========================================="
echo "所有Exchange Rate MPS实验完成!"
echo "==========================================" 
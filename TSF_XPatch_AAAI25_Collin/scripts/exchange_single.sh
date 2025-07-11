#!/bin/bash

# Single Exchange Rate Experiment
# Quick test for Exchange Rate dataset

echo "=========================================="
echo "Exchange Rate Dataset - Single Experiment"
echo "=========================================="

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 创建日志目录
mkdir -p ./logs/ema

# 提示用户激活环境
echo "注意: 请确保已经激活xpatch环境:"
echo "conda activate xpatch"
echo ""

# 参数说明：
# --is_training 1: 训练模式 (1=训练, 0=测试)
# --root_path: 数据集根目录路径
# --data_path: 具体数据文件路径
# --model_id: 模型实验ID，用于保存结果
# --model: 使用的模型名称 (xPatch)
# --data: 数据加载器类型 (custom表示自定义数据集)
# --features: 特征类型 (M=多变量, S=单变量, MS=多变量预测单变量)
# --seq_len: 输入序列长度 (历史时间步数)
# --pred_len: 预测序列长度 (未来时间步数)
# --enc_in: 编码器输入特征维度 (8种汇率)
# --des: 实验描述标识符
# --itr: 实验重复次数 (取平均值)
# --batch_size: 批大小 (32适合macOS内存)
# --learning_rate: 学习率 (Adam优化器)
# --lradj: 学习率调度策略 (sigmoid衰减)
# --ma_type: 移动平均类型 (ema=指数移动平均, dema=双指数)
# --alpha: EMA的alpha参数 (平滑系数)
# --beta: DEMA的beta参数 (当ma_type=dema时使用)
# --num_workers: 数据加载进程数 (macOS设为0避免multiprocessing问题)
# --use_gpu: 启用GPU加速 (自动选择MPS/CUDA/CPU)
# --train_epochs: 训练轮数 (快速测试用2轮)
# --patience: 早停耐心值 (连续3轮无改善则停止)

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate/exchange_rate.csv \
  --model_id exchange_96_ema_single \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.00001 \
  --lradj 'sigmoid' \
  --ma_type ema \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0 \
  --use_gpu True \
#   --train_epochs 2 \
#   --patience 3

echo "=========================================="
echo "Exchange Rate experiment completed!"
echo "==========================================" 
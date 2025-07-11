#!/bin/bash

# Exchange Rate Dataset Experiment Script
# Based on xPatch unified settings

ma_type=ema
alpha=0.3
beta=0.3

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 创建日志目录
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/"$ma_type ]; then
    mkdir ./logs/$ma_type
fi

model_name=xPatch
seq_len=96

echo "=========================================="
echo "Starting Exchange Rate Dataset Experiments"
echo "Model: $model_name"
echo "Moving Average Type: $ma_type"
echo "Sequence Length: $seq_len"
echo "=========================================="

# 针对不同预测长度进行实验
for pred_len in 96 192 336 720
do
  echo "Running experiment: pred_len=$pred_len"
  
  python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path exchange_rate/exchange_rate.csv \
    --model_id exchange_${pred_len}_${ma_type}_unified \
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
    --train_epochs 100 \
    --patience 10 > logs/$ma_type/${model_name}_exchange_${seq_len}_${pred_len}_unified.log
    
  echo "Completed: pred_len=$pred_len"
  echo "Log saved to: logs/$ma_type/${model_name}_exchange_${seq_len}_${pred_len}_unified.log"
  echo "------------------------------------------"
done

echo "=========================================="
echo "All Exchange Rate experiments completed!"
echo "Check logs directory for detailed results."
echo "==========================================" 
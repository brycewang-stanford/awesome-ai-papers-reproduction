# xPatch Experiments on Exchange Rate Dataset

## Overview

This document summarizes the successful experiments conducted on the exchange_rate dataset using the xPatch model for time series forecasting. All experiments were conducted following the fair configuration settings as specified in the original paper.

## Experiment Configuration

### Model Settings
- **Model**: xPatch (Patch-based Time Series Forecasting)
- **Dataset**: exchange_rate.csv
- **Features**: Multivariate (M)
- **Input Features**: 8 channels (enc_in=8)
- **Sequence Length**: 96 (seq_len=96)
- **Label Length**: 48 (label_len=48)
- **Moving Average Type**: Regular (ma_type=reg)
- **Alpha**: 0.3
- **Beta**: 0.3

### Training Configuration
- **Batch Size**: 32
- **Learning Rate**: 0.0001
- **Learning Rate Adjustment**: type1
- **Maximum Epochs**: 10
- **Early Stopping Patience**: 3
- **Loss Function**: MSE
- **Optimization**: Adam (default)
- **Device**: CPU (automatically detected)

### Data Preprocessing
- **Normalization**: RevIN (Reversible Instance Normalization)
- **Patch Length**: 16
- **Stride**: 8
- **Padding**: End padding

## Executed Experiments

### 1. Prediction Length 96

**Bash Command:**
```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id exchange_96_reg \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --train_epochs 10 \
  --patience 3 \
  --ma_type reg \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0
```

**Results:**
- MSE: 0.0807
- MAE: 0.1972

### 2. Prediction Length 192

**Bash Command:**
```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id exchange_192_reg \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --train_epochs 10 \
  --patience 3 \
  --ma_type reg \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0
```

**Results:**
- MSE: 0.1748
- MAE: 0.2960

### 3. Prediction Length 336

**Bash Command:**
```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id exchange_336_reg \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --train_epochs 10 \
  --patience 3 \
  --ma_type reg \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0
```

**Results:**
- MSE: 0.3434
- MAE: 0.4213

### 4. Prediction Length 720

**Bash Command:**
```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id exchange_720_reg \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --train_epochs 10 \
  --patience 3 \
  --ma_type reg \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0
```

**Results:**
- MSE: 0.8568
- MAE: 0.6976

## Results Summary

### Performance Metrics

| Prediction Length | MSE | MAE | Dataset Split (Train/Val/Test) |
|-------------------|-----|-----|--------------------------------|
| 96 | 0.0807 | 0.1972 | 4880/425/1182 |
| 192 | 0.1748 | 0.2960 | 4880/425/1182 |
| 336 | 0.3434 | 0.4213 | 4880/425/1182 |
| 720 | 0.8568 | 0.6976 | 4496/41/798 |

### Statistical Analysis

- **Average MSE**: 0.3640
- **Average MAE**: 0.4030
- **MSE Standard Deviation**: 0.2997
- **MAE Standard Deviation**: 0.1877

### Performance Degradation Analysis

Performance degradation relative to pred_len=96:

| Prediction Length | MSE Increase | MAE Increase |
|-------------------|--------------|--------------|
| 192 | +116.6% | +50.1% |
| 336 | +325.4% | +113.6% |
| 720 | +961.2% | +253.7% |

## Key Observations

1. **Short-term Prediction Excellence**: The model performs exceptionally well on short-term predictions (96 steps).
2. **Performance Degradation**: There's a clear degradation in performance as the prediction horizon increases.
3. **Non-linear Decay**: The performance drop is not linear; longer predictions show exponential degradation.
4. **Data Split Variation**: The 720-step prediction uses a different data split due to sequence length constraints.

## Environment Setup

### Virtual Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_fixed.txt
```

### Key Dependencies
- torch
- numpy
- pandas
- scikit-learn
- matplotlib

### Compatibility Fixes Applied
- Fixed `np.Inf` to `np.inf` for NumPy 2.0 compatibility
- Modified device handling from hardcoded `'cuda'` to automatic device detection
- Set `num_workers=0` to avoid multiprocessing issues on macOS

## File Locations

- **Results**: `result.txt`
- **Checkpoints**: `./checkpoints/`
- **Logs**: `./logs/`
- **Scripts**: `./scripts/xPatch_fair.sh`

---

# xPatch 在汇率数据集上的实验总结

## 概述

本文档总结了使用xPatch模型在exchange_rate数据集上进行时间序列预测的成功实验。所有实验都遵循原论文中指定的公平配置设置。

## 实验配置

### 模型设置
- **模型**: xPatch（基于补丁的时间序列预测）
- **数据集**: exchange_rate.csv
- **特征**: 多变量 (M)
- **输入特征**: 8个通道 (enc_in=8)
- **序列长度**: 96 (seq_len=96)
- **标签长度**: 48 (label_len=48)
- **移动平均类型**: 常规 (ma_type=reg)
- **Alpha**: 0.3
- **Beta**: 0.3

### 训练配置
- **批大小**: 32
- **学习率**: 0.0001
- **学习率调整**: type1
- **最大轮次**: 10
- **早停耐心**: 3
- **损失函数**: MSE
- **优化器**: Adam (默认)
- **设备**: CPU (自动检测)

### 数据预处理
- **归一化**: RevIN (可逆实例归一化)
- **补丁长度**: 16
- **步长**: 8
- **填充**: 末尾填充

## 执行的实验

### 1. 预测长度 96

**Bash命令:**
```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id exchange_96_reg \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --train_epochs 10 \
  --patience 3 \
  --ma_type reg \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0
```

**结果:**
- MSE: 0.0807
- MAE: 0.1972

### 2. 预测长度 192

**Bash命令:**
```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id exchange_192_reg \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --train_epochs 10 \
  --patience 3 \
  --ma_type reg \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0
```

**结果:**
- MSE: 0.1748
- MAE: 0.2960

### 3. 预测长度 336

**Bash命令:**
```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id exchange_336_reg \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --train_epochs 10 \
  --patience 3 \
  --ma_type reg \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0
```

**结果:**
- MSE: 0.3434
- MAE: 0.4213

### 4. 预测长度 720

**Bash命令:**
```bash
python run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --model_id exchange_720_reg \
  --model xPatch \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --lradj 'type1' \
  --train_epochs 10 \
  --patience 3 \
  --ma_type reg \
  --alpha 0.3 \
  --beta 0.3 \
  --num_workers 0
```

**结果:**
- MSE: 0.8568
- MAE: 0.6976

## 结果总结

### 性能指标

| 预测长度 | MSE | MAE | 数据集分割 (训练/验证/测试) |
|---------|-----|-----|---------------------------|
| 96 | 0.0807 | 0.1972 | 4880/425/1182 |
| 192 | 0.1748 | 0.2960 | 4880/425/1182 |
| 336 | 0.3434 | 0.4213 | 4880/425/1182 |
| 720 | 0.8568 | 0.6976 | 4496/41/798 |

### 统计分析

- **平均MSE**: 0.3640
- **平均MAE**: 0.4030
- **MSE标准差**: 0.2997
- **MAE标准差**: 0.1877

### 性能下降分析

相对于pred_len=96的性能下降：

| 预测长度 | MSE增加 | MAE增加 |
|---------|---------|---------|
| 192 | +116.6% | +50.1% |
| 336 | +325.4% | +113.6% |
| 720 | +961.2% | +253.7% |

## 关键观察

1. **短期预测优势**: 模型在短期预测（96步）上表现出色。
2. **性能下降**: 随着预测范围增加，性能明显下降。
3. **非线性衰减**: 性能下降不是线性的；更长的预测显示指数级衰减。
4. **数据分割变化**: 720步预测由于序列长度限制使用了不同的数据分割。

## 环境配置

### 虚拟环境
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements_fixed.txt
```

### 关键依赖
- torch
- numpy
- pandas
- scikit-learn
- matplotlib

### 应用的兼容性修复
- 将`np.Inf`修复为`np.inf`以兼容NumPy 2.0
- 将设备处理从硬编码`'cuda'`修改为自动设备检测
- 设置`num_workers=0`以避免在macOS上的多进程问题

## 文件位置

- **结果**: `result.txt`
- **检查点**: `./checkpoints/`
- **日志**: `./logs/`
- **脚本**: `./scripts/xPatch_fair.sh`

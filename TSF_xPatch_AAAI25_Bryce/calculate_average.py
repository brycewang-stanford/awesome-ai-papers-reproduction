#!/usr/bin/env python3
"""
计算exchange_rate数据集实验结果的平均值
"""

# exchange_rate数据集的实验结果
results = {
    'exchange_96_reg': {'mse': 0.08073386549949646, 'mae': 0.19722530245780945},
    'exchange_192_reg': {'mse': 0.1748499721288681, 'mae': 0.29598572850227356},
    'exchange_336_reg': {'mse': 0.34343045949935913, 'mae': 0.421256422996521},
    'exchange_720_reg': {'mse': 0.856787919998169, 'mae': 0.6975865364074707}
}

print("="*60)
print("exchange_rate数据集实验结果汇总")
print("="*60)

print("\n详细结果:")
for exp_name, metrics in results.items():
    pred_len = exp_name.split('_')[1]
    print(f"pred_len={pred_len:>3}: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}")

# 计算平均值
mse_values = [metrics['mse'] for metrics in results.values()]
mae_values = [metrics['mae'] for metrics in results.values()]

avg_mse = sum(mse_values) / len(mse_values)
avg_mae = sum(mae_values) / len(mae_values)

print("\n" + "="*60)
print("平均值计算:")
print("="*60)
print(f"MSE平均值: {avg_mse:.4f}")
print(f"MAE平均值: {avg_mae:.4f}")

# 计算标准差
import math

mse_std = math.sqrt(sum((x - avg_mse)**2 for x in mse_values) / len(mse_values))
mae_std = math.sqrt(sum((x - avg_mae)**2 for x in mae_values) / len(mae_values))

print(f"\nMSE标准差: {mse_std:.4f}")
print(f"MAE标准差: {mae_std:.4f}")

# 最佳和最差结果
min_mse_idx = mse_values.index(min(mse_values))
max_mse_idx = mse_values.index(max(mse_values))
min_mae_idx = mae_values.index(min(mae_values))
max_mae_idx = mae_values.index(max(mae_values))

exp_names = list(results.keys())
print(f"\n最佳MSE: {min(mse_values):.4f} ({exp_names[min_mse_idx]})")
print(f"最差MSE: {max(mse_values):.4f} ({exp_names[max_mse_idx]})")
print(f"最佳MAE: {min(mae_values):.4f} ({exp_names[min_mae_idx]})")
print(f"最差MAE: {max(mae_values):.4f} ({exp_names[max_mae_idx]})")

# 性能下降分析
print("\n" + "="*60)
print("性能下降分析 (相对于pred_len=96):")
print("="*60)
base_mse = results['exchange_96_reg']['mse']
base_mae = results['exchange_96_reg']['mae']

for exp_name, metrics in results.items():
    if exp_name == 'exchange_96_reg':
        continue
    pred_len = exp_name.split('_')[1]
    mse_increase = (metrics['mse'] - base_mse) / base_mse * 100
    mae_increase = (metrics['mae'] - base_mae) / base_mae * 100
    print(f"pred_len={pred_len:>3}: MSE增加{mse_increase:>6.1f}%, MAE增加{mae_increase:>6.1f}%")

print("\n" + "="*60)
print("计算完成!")
print("="*60)

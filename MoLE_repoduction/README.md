# MoLE (AISTATS 2024)

This is the official implementation of the paper "Mixture-of-Linear-Experts for Long-term Time Series Forecasting". [[arXiv]](https://arxiv.org/abs/2312.06786) [[PMLR]](https://proceedings.mlr.press/v238/ni24a.html)

## Reproduction of MoLE Paper Results

This repository now includes scripts to reproduce the results of the MoLE paper on the electricity dataset (ECL.csv). The following scripts have been added:

1. **`evaluate_mole.py`**: Evaluates the MoLE model on the electricity dataset, printing metrics (MAE, MSE, RMSE, etc.) and providing prediction/ground truth statistics.
2. **`run_prediction.py`**: Runs MoLE model predictions, saves results to the `./results/` directory, and prints out generated files and their statistics.
3. **`show_mae_results.py`**: Loads and displays MAE and other metrics from the results folder, summarizes them in a DataFrame, and saves metrics and data summaries as CSV files.
4. **`simple_eval.py`**: Evaluates baseline prediction strategies (last value, trend, moving average, linear regression) on the ECL dataset, computes metrics, and compares results.

### One Epoch Training

A demonstration training script has been added for quick experimentation:

5. **`train_mole_dlinear_ecl_one_epoch.sh`**: Trains the MoLE DLinear model on the ECL dataset for one epoch with pre-configured parameters. This script demonstrates:
   - Training parameters: seq_len=336, pred_len=96, learning_rate=0.0001, t_dim=4, batch_size=8
   - Model checkpoints saved to `./checkpoints/one_epoch_model/`
   - Training logs saved to `./logs/one_epoch_training/`
   - Achieves validation loss of 0.126 and test loss of 0.146 after one epoch

These scripts are designed for user-friendly analysis and can be executed directly to replicate the findings of the paper.

---

# 复现 MoLE 论文结果

本仓库现已包含复现 MoLE 论文在电力数据集（ECL.csv）上的结果的脚本。新增的脚本如下：

1. **`evaluate_mole.py`**: 评估 MoLE 模型在电力数据集上的表现，打印指标（MAE, MSE, RMSE 等）并提供预测/真实值的统计信息。
2. **`run_prediction.py`**: 运行 MoLE 模型预测，将结果保存到 `./results/` 目录，并打印生成文件及其统计信息。
3. **`show_mae_results.py`**: 加载并显示结果文件夹中的 MAE 和其他指标，将其汇总为 DataFrame，并将指标和数据摘要保存为 CSV 文件。
4. **`simple_eval.py`**: 评估基线预测策略（最后值、趋势、移动平均、线性回归）在 ECL 数据集上的表现，计算指标并比较结果。

### 单轮训练

已新增演示训练脚本用于快速实验：

5. **`train_mole_dlinear_ecl_one_epoch.sh`**: 在 ECL 数据集上训练 MoLE DLinear 模型一轮，使用预配置参数。该脚本演示：
   - 训练参数：seq_len=336，pred_len=96，learning_rate=0.0001，t_dim=4，batch_size=8
   - 模型检查点保存到 `./checkpoints/one_epoch_model/`
   - 训练日志保存到 `./logs/one_epoch_training/`
   - 一轮训练后获得验证损失 0.126 和测试损失 0.146

这些脚本设计为用户友好型，可直接执行以复现论文的研究结果。

---

## Requirements

Please refer to the `requirements.txt` file for the required packages.

## Datasets

All datasets we used in our experiments (except Weather2K) are available at this [Google Drive's shared folder](https://drive.google.com/drive/folders/1ZhaQwLYcnhT5zEEZhBTo03-jDwl7bt7v). These datasets were first provided in Autoformer. Please download the datasets and put them in the `dataset` folder. Each dataset is an `.csv` file.

Weather2K dataset is available at this [GitHub repository](https://github.com/bycnfz/weather2k).

## Usage

### Main Experiments

To run the main experiments, please run the following command:

```bash
scripts/run_all_3_seeds.sh
```

This scripts sequentially runs the main experiments on all datasets with 3 different random seeds. The results will be saved in the `logs` folder.

### Additional Experiments

The repository has been cleaned up for easier usage. If you want to run ablation experiments, please refer to earlier commits.

## Acknowledgement

We thank the authors of the following repositories for their open-source code or dataset, which we used in our experiments:

https://github.com/zhouhaoyi/Informer2020

https://github.com/cure-lab/LTSF-Linear

https://github.com/plumprc/RTSF

https://github.com/bycnfz/weather2k

## Citation
If you find our work useful, please consider citing our paper using the following BibTeX:
```
@inproceedings{ni2024mixture,
  title={Mixture-of-Linear-Experts for Long-term Time Series Forecasting},
  author={Ni, Ronghao and Lin, Zinan and Wang, Shuaiqi and Fanti, Giulia},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={4672--4680},
  year={2024},
  organization={PMLR}
}
```

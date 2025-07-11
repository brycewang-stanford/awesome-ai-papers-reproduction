# Awesome AI Papers Reproduction

[English](#englis2. **Navigate to a specific reproduction**:
```bash
cd T```
awesome-AI-papers-#```
awesome-AI-papers-reproduction/
├── TSF_MoLE_AISTATS24_Bryce/   # 线性专家混合模型复现
│   ├── models/                 # 核心模型实现
│   ├── data_provider/          # 数据加载和预处理
│   ├── scripts/               # 训练和评估脚本
│   ├── results/               # 实验结果
│   └── README.md              # 具体复现文档
├── TSF_XPatch_AAAI25_Collin/  # XPatch 论文复现 (Collin版本)
├── TSF_XPatch_AAAI25_Jerry/   # XPatch 论文复现 (Jerry版本)
└── [未来的复现项目...]
```E (线性专家2. **进入特定复现项目**:
```bash
cd TSF_MoLE_AISTATS24型)
- **状态**: ✅ 初始实现完成
- **数据集**: ECL (电力消耗负荷)
- **模型**: DLinear, RLinear, RMLP 变体
- **结果**: 可在 `TSF_MoLE_AISTATS24/results/` 中查看duction/
├── TSF_MoLE_AISTATS24/         # 线性专家混合模型复现
│   ├── models/                 # 核心模型实现
│   ├── data_provider/          # 数据加载和预处理
│   ├── scripts/               # 训练和评估脚本
│   ├── results/               # 实验结果
│   └── README.md              # 具体复现文档
└── [未来的复现项目...]
```STATS24| [中文](#中文)

---

## English

### Overview

This repository focuses on the rapid reproduction of papers from top AI conferences including NIPS/NeurIPS, ICML, ICLR, and AAAI. Our goal is to provide efficient implementations of cutting-edge research with practical evaluation frameworks.

### Methodology

Our reproduction process follows a systematic approach:

1. **Initial Implementation**: We start by implementing the core model using the simplest possible dataset to validate the fundamental concepts and architecture.

2. **Training & Evaluation**: Each model is trained and evaluated on baseline datasets to ensure correctness of implementation.

3. **Full Reproduction** (when time and resources permit): If initial results are promising and resources allow, we proceed with complete reproduction using the original datasets and experimental setups.

### Repository Structure

```
awesome-AI-papers-reproduction/
├── TSF_MoLE_AISTATS24_Bryce/   # Mixture of Linear Experts reproduction
│   ├── models/                 # Core model implementations
│   ├── data_provider/          # Data loading and preprocessing
│   ├── scripts/               # Training and evaluation scripts
│   ├── results/               # Experimental results
│   └── README.md              # Specific reproduction documentation
├── TSF_XPatch_AAAI25_Collin/  # XPatch paper reproduction (Collin's version)
├── TSF_XPatch_AAAI25_Jerry/   # XPatch paper reproduction (Jerry's version)
└── [Future reproductions...]
```

### Current Reproductions

#### 1. MoLE (Mixture of Linear Experts)
- **Status**: ✅ Initial implementation complete
- **Dataset**: ECL (Electricity Consuming Load)
- **Models**: DLinear, RLinear, RMLP variants
- **Results**: Available in `TSF_MoLE_AISTATS24_Bryce/results/`

#### 2. XPatch (Time Series Forecasting)
- **Status**: 📄 Paper available for reproduction
- **Venue**: AAAI 2025
- **Contributors**: Collin & Jerry versions
- **Document**: Available in respective directories

### Getting Started

1. **Clone the repository**:
```bash
git clone https://github.com/brycewang-stanford/awesome-ai-papers-reproduction.git
cd awesome-ai-papers-reproduction
```

2. **Navigate to specific reproduction**:
```bash
cd TSF_MoLE_AISTATS24_Bryce
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Run experiments**:
```bash
# Quick evaluation
python simple_eval.py

# Full training
bash scripts/MoE_ECL.sh
```

### Contribution Guidelines

- Each paper reproduction should have its own directory
- Include clear documentation of implementation choices
- Provide both quick demo and full reproduction scripts
- Document any deviations from the original paper
- Include evaluation metrics and comparison with paper results

### Research Focus Areas

- **Time Series Forecasting**: MoLE, Transformers, Linear models
- **Computer Vision**: [Planned reproductions]
- **Natural Language Processing**: [Planned reproductions]
- **Reinforcement Learning**: [Planned reproductions]

---

## 中文

### 概述

本仓库专注于快速复现顶级AI会议论文，包括NIPS/NeurIPS、ICML、ICLR和AAAI等。我们的目标是提供前沿研究的高效实现和实用的评估框架。

### 方法论

我们的复现过程遵循系统化方法：

1. **初始实现**：使用最简单的数据集实现核心模型，验证基本概念和架构。

2. **训练与评估**：在基准数据集上训练和评估每个模型，确保实现的正确性。

3. **完整复现**（时间和资源允许时）：如果初始结果有希望且资源允许，我们会使用原始数据集和实验设置进行完整复现。

### 仓库结构

```
awesome-AI-papers-reproduction/
├── MoLE_reproduction/          # 线性专家混合模型复现
│   ├── models/                 # 核心模型实现
│   ├── data_provider/          # 数据加载和预处理
│   ├── scripts/               # 训练和评估脚本
│   ├── results/               # 实验结果
│   └── README.md              # 特定复现文档
└── [未来的复现项目...]
```

### 当前复现项目

#### 1. MoLE (线性专家混合模型)
- **状态**: ✅ 初始实现完成
- **数据集**: ECL (电力消耗负载)
- **模型**: DLinear, RLinear, RMLP 变体
- **结果**: 可在 `TSF_MoLE_AISTATS24_Bryce/results/` 中查看

#### 2. XPatch (时间序列预测)
- **状态**: 📄 论文可供复现
- **会议**: AAAI 2025
- **贡献者**: Collin & Jerry 版本
- **文档**: 在各自目录中可用

### 快速开始

1. **克隆仓库**:
```bash
git clone https://github.com/brycewang-stanford/awesome-ai-papers-reproduction.git
cd awesome-ai-papers-reproduction
```

2. **进入特定复现目录**:
```bash
cd TSF_MoLE_AISTATS24_Bryce
```

3. **安装依赖**:
```bash
pip install -r requirements.txt
```

4. **运行实验**:
```bash
# 快速评估
python simple_eval.py

# 完整训练
bash scripts/MoE_ECL.sh
```

### 贡献指南

- 每个论文复现应有自己的目录
- 包含实现选择的清晰文档
- 提供快速演示和完整复现脚本
- 记录与原论文的任何偏差
- 包含评估指标和与论文结果的比较

### 研究重点领域

- **时间序列预测**: MoLE, Transformers, 线性模型
- **计算机视觉**: [计划中的复现]
- **自然语言处理**: [计划中的复现]
- **强化学习**: [计划中的复现]

---

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments

- Thanks to the original authors of all reproduced papers
- Inspired by the open science movement in AI research
- Special thanks to the research community for making code and data available

### Contact

For questions or collaboration opportunities, please open an issue or contact the maintainers.
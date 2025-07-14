# 新手上路：抑郁症检测论文复现 (Depression Detection Paper Reproduction)

## 中文说明

### 项目简介
本项目是对期刊论文《An attention-based hybrid architecture with explainability for depressive social media text detection in Bangla》的复现工作。论文提出了一种基于注意力机制的混合架构，用于检测社交媒体文本中的抑郁倾向，并提供可解释性。该项目适合新手上手，帮助理解基础的深度学习模型和数据处理流程。

### 数据集
- **Twitter 数据集**: 包含推文及其对应的抑郁倾向标签。
- **Reddit 数据集**: 包含从 Reddit 平台收集的文本及其分类标签。
- **Facebook 数据集**: 包含负样本的文本数据。

数据集存储在 `dataset/` 文件夹下，分为 `Twitter/`、`Reddit/` 和 `Facebook/` 子文件夹。

### 环境依赖
请确保安装以下依赖：
- Python 3.8 或更高版本
- TensorFlow
- NLTK
- Gensim
- Scikit-learn
- Matplotlib

可以通过以下命令安装依赖：
```bash
pip install -r requirements.txt
```

### 运行步骤
1. 下载并准备数据集，将其放置在 `dataset/` 文件夹中。
2. 运行 Jupyter Notebook 文件 `An_attention_based_hybrid_architecture_done.ipynb`，按照代码中的步骤完成数据预处理、模型训练和评估。
3. 训练完成后，模型将保存为 `trained_model_Reddit_Facebook.h5`。

### 结果
- 模型在测试集上的准确率、精确率、召回率和 F1 分数均达到了较高水平。
- 详细的评估结果可以在 Notebook 的输出中查看。

---

## English Description

### Start from here: Project Overview
This project reproduces a journal paper "An attention-based hybrid architecture with explainability for depressive social media text detection in Bangla." The paper proposes an attention-based hybrid architecture for detecting depressive tendencies in social media texts with explainability. This project is beginner-friendly and helps in understanding the basics of deep learning models and data processing workflows.

### Datasets
- **Twitter Dataset**: Contains tweets and their corresponding depression tendency labels.
- **Reddit Dataset**: Contains texts collected from the Reddit platform and their classification labels.
- **Facebook Dataset**: Contains negative sample text data.

The datasets are stored in the `dataset/` folder, divided into `Twitter/`, `Reddit/`, and `Facebook/` subfolders.

### Environment Requirements
Ensure the following dependencies are installed:
- Python 3.8 or higher
- TensorFlow
- NLTK
- Gensim
- Scikit-learn
- Matplotlib

Install dependencies using the following command:
```bash
pip install -r requirements.txt
```

### Steps to Run
1. Download and prepare the datasets, placing them in the `dataset/` folder.
2. Run the Jupyter Notebook file `An_attention_based_hybrid_architecture_done.ipynb` and follow the steps in the code to complete data preprocessing, model training, and evaluation.
3. After training, the model will be saved as `trained_model_Reddit_Facebook.h5`.

### Results
- The model achieved high accuracy, precision, recall, and F1 scores on the test set.
- Detailed evaluation results can be found in the Notebook's output.

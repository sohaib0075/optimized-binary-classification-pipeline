# 🤖 CPU vs GPU ML Model Benchmarking

> **Team:** 22i-2034 · 22i-1873 · 22i-2033

A comparative study of five machine learning models on a binary classification task, benchmarking CPU and GPU processing in terms of accuracy, F1 score, training time, and computational speedup.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset & Preprocessing](#dataset--preprocessing)
- [Models](#models)
- [Results](#results)
- [Speedup Analysis](#speedup-analysis)
- [Installation](#installation)
- [Usage](#usage)
- [Conclusion](#conclusion)

---

## Overview

This project evaluates five machine learning models on a balanced binary classification dataset, comparing their predictive performance alongside computational efficiency across CPU and GPU execution environments.

**Key Questions:**
- Which model achieves the best accuracy and F1 score?
- How much faster are parallel/GPU models compared to serial CPU execution?
- What are the trade-offs between speed and performance?

---

## Dataset & Preprocessing

**Features:** 8 columns — `feature_1` through `feature_7` + `target`

### Preprocessing Steps

| Step | Method |
|------|--------|
| Missing Values (numerical) | Replaced with **median** |
| Missing Values (categorical) | Replaced with **mode** |
| Encoding | Label Encoding for categorical columns |
| Normalization | `StandardScaler` |
| Class Imbalance | **SMOTE** (Synthetic Minority Oversampling) |
| Train/Test Split | 80% training / 20% testing |

---

## Models

Five models were trained and evaluated:

| # | Model | Hardware |
|---|-------|----------|
| 1 | Serial Random Forest | CPU (single-threaded) |
| 2 | Parallel Random Forest | CPU (multi-threaded) |
| 3 | PyTorch Neural Network | GPU |
| 4 | XGBoost Classifier | GPU |
| 5 | CatBoost Classifier | CPU |

---

## Results

### Performance Summary

| Model | Accuracy | F1 Score | Training Time |
|-------|----------|----------|---------------|
| **Serial RF** | **0.6322** | **0.6205** | 7.22s |
| **Parallel RF** | **0.6322** | **0.6205** | 1.10s |
| PyTorch NN | 0.5042 | 0.4285 | 3.25s |
| XGBoost | 0.5694 | 0.5564 | 1.04s |
| **CatBoost** | **0.6421** | 0.5426 | 2.98s |

🏆 **Best Accuracy:** CatBoost (0.6421)  
🏆 **Best F1 Score:** Random Forest — Serial & Parallel (0.6205)  
⚡ **Fastest Training:** XGBoost (1.04s)

---

## Speedup Analysis

Speedup calculated relative to the Serial Random Forest baseline (7.22s):

| Model | Training Time | Speedup vs Serial RF |
|-------|--------------|----------------------|
| Serial RF | 7.22s | — (baseline) |
| Parallel RF | 1.10s | 84.8% faster |
| PyTorch (GPU) | 3.25s | 54.99% faster |
| XGBoost (GPU) | 1.04s | 85.63% faster |
| CatBoost | 2.98s | 58.74% faster |

### CPU vs GPU Overall

| Metric | Value |
|--------|-------|
| Total CPU Models Time | 12.33s |
| Total GPU Model Time (PyTorch) | 3.25s |
| **GPU Speedup over CPU** | **73.66%** |

> GPU models are significantly faster but didn't outperform CPU-based Random Forests on accuracy/F1 for this task.

---

## Installation

```bash
git clone https://github.com/<your-username>/ml-cpu-gpu-benchmark.git
cd ml-cpu-gpu-benchmark
pip install -r requirements.txt
```

**Dependencies:**

```
scikit-learn
xgboost
catboost
torch
imbalanced-learn   # for SMOTE
pandas
numpy
matplotlib
```

---

## Usage

### Preprocess Data

```bash
python preprocess.py --input data/dataset.csv --output data/processed.csv
```

### Train All Models

```bash
python train.py --all
```

### Train a Specific Model

```bash
python train.py --model serial_rf
python train.py --model parallel_rf
python train.py --model pytorch
python train.py --model xgboost
python train.py --model catboost
```

### Evaluate & Compare

```bash
python evaluate.py --report
```

---

## Conclusion

This project demonstrates clear trade-offs between training speed and predictive performance:

- **Random Forests (CPU)** delivered the most reliable F1 scores, with the parallel version achieving the same accuracy as serial at ~6.6× the speed.
- **GPU-accelerated models** (PyTorch, XGBoost) were significantly faster but underperformed on accuracy and F1 for this dataset.
- **CatBoost** achieved the highest accuracy (0.6421) but with a lower F1 (0.5426), suggesting it favors one class.
- **PyTorch NN** struggled most, barely exceeding random-chance accuracy (0.5042), indicating the dataset may not suit deep learning without more tuning.

**Key Takeaway:** Hardware acceleration does not guarantee better results — model choice should be guided by task characteristics, dataset size, and performance requirements.

---

## Project Info

| Field | Detail |
|-------|--------|
| Team Members | 22i-2034, 22i-1873, 22i-2033 |
| Task | Binary Classification Benchmarking |
| Best F1 Model | Random Forest (Serial & Parallel) — F1: 0.6205 |
| Best Accuracy Model | CatBoost — Accuracy: 0.6421 |
| Fastest Model | XGBoost — 1.04s |

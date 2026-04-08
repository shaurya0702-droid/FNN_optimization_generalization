# Deep Learning

A collection of hands-on deep learning projects built with PyTorch, exploring foundational architectures and training dynamics from the ground up.

---

## Projects

### 1. FNN — Optimization & Generalization on MNIST

> *How much does your optimizer actually matter? More than your architecture.*

A structured experimental study of Feedforward Neural Networks on MNIST. Rather than just training a classifier, this project systematically investigates how optimization strategies, model capacity, learning rate, normalization, and regularization each independently influence convergence and generalization.

**Architecture:** `Input (784) → Hidden Layers (variable) → ReLU → Output (10)`

**Experiments (one variable changed at a time):**

| Experiment | Variants | Best Result |
|---|---|---|
| Optimizer | SGD, SGD+Momentum, Adam | Adam → ~89.9% |
| Model Capacity | `[64]`, `[128]`, `[256, 128]` | Large → ~88.8% |
| Learning Rate | 0.1, 0.01, 0.001 | 0.01 → ~89% |
| Normalization | With / Without | Without → ~89.3% |
| Regularization (L2) | None / 1e-4 | None → ~89.6% |

**Key findings:**
- Optimizer choice has the largest single impact on convergence speed
- Normalization and regularization don't always help — MNIST is too simple to benefit
- High learning rate (0.1) causes divergence; 0.01 is near-optimal for this setup
- Deep learning performance is governed as much by training dynamics as by architecture

📁 `FNN_optimization_generalization/`

---

### 2. CNN — Image Classification on CIFAR-10

> *A clean, end-to-end deep learning pipeline achieving ~82% test accuracy.*

A convolutional neural network for 10-class image classification on CIFAR-10, trained on a GPU (Google Colab T4). Demonstrates progressive channel expansion, batch normalization, dropout regularization, and data augmentation — without any learning rate scheduling or advanced tricks.

**Architecture:**

| Stage | Details |
|---|---|
| Conv Block ×4 | 3×3 Conv → BatchNorm → ReLU → MaxPool |
| Channel progression | 3 → 64 → 128 → 256 → 512 |
| Spatial progression | 32 → 16 → 8 → 4 → 2 |
| Classifier | 2048 → 256 → Dropout(0.5) → 10 |

**Training config:** Adam (lr=0.0007) · CrossEntropyLoss · Batch size 256 · 15 epochs

**Results:**

| Metric | Value |
|---|---|
| Training Accuracy | ~83% |
| Test Accuracy | ~82% |

The narrow train/test gap indicates well-controlled overfitting, attributable to batch normalization and augmentation. Strong performance relative to the architectural simplicity.

📁 `CNN_CIFAR-10/`

---

### 3. RNN vs LSTM vs GRU — Recursive Time Series Extrapolation

> *Short-term prediction is easy. Cut off the real data — that's where the difference shows up.*

All three models are trained on a sine-like wave and evaluated on **recursive extrapolation**: given a seed window, the model generates the next 100 steps using only its own predictions as input. This stress-tests memory retention in a way standard evaluation never does.

**Signal:** `y = sin(t + cos(t)) + noise (σ=0.03), t ∈ [0, 30π], N=500`

**Controlled setup** — every hyperparameter is identical across models:

| Setting | Value |
|---|---|
| Hidden size | 12 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Epochs | 50 |
| Input window | 20 steps |

**Results:**

| Model | Short-term prediction | Recursive extrapolation (100 steps) |
|---|---|---|
| RNN | ✅ Tracks signal | ❌ Drifts and flattens |
| LSTM | ✅ Tracks signal | ✅ Holds the wave |
| GRU | ✅ Tracks signal | ✅ Holds the wave |

A plain RNN rewrites its hidden state at every step with no gating — small prediction errors compound until the memory collapses to a fixed point. LSTM and GRU use learned gates to decide what to keep and what to discard, which is the only reason they sustain the signal. Everything else was identical.

📁 `RNN_LSTM_GRU_Timeseries/`

---

## Repository Structure

```
DeepLearning/
├── FNN_optimization_generalization/
│   ├── fnn_optimization_generalization.ipynb
│   └── README.md
├── CNN_CIFAR-10/
│   ├── CNN_CIFAR_10.ipynb
│   ├── cnn_cifar_10.py
│   └── README.md
├── RNN_LSTM_GRU_Timeseries/
│   ├── rnn_lstm_gru.ipynb
│   └── README.md
└── README.md
```

---

## Getting Started

**Prerequisites**
```bash
pip install torch torchvision numpy matplotlib
```

Each project is self-contained. Clone the repo and open the notebook for any project:

```bash
git clone <your-repo-url>
cd DeepLearning
jupyter notebook
```

For the CNN project, a GPU is recommended. If using Google Colab, enable T4 GPU under **Runtime → Change runtime type**.

---

## Tech Stack

Python · PyTorch · Torchvision · NumPy · Sklearn · Matplotlib

---

REFERENCE: A deep understanding of deep learning  ~ Mike X Cohen, sincxpress.com

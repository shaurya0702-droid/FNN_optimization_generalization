# CIFAR-10 Image Classification using CNN (PyTorch)
 
## Overview
 
This project implements a convolutional neural network for multi-class image classification on the CIFAR-10 dataset using PyTorch. The model is trained on a GPU (Google Colab T4) and demonstrates a clean, end-to-end deep learning pipeline covering data loading, augmentation, model definition, training, and evaluation.
 
The architecture follows a standard deep CNN design with progressive channel expansion, batch normalization, and dropout regularization, achieving approximately 82% test accuracy without learning rate scheduling or advanced architectural tricks.
 
---
 
## Features
 
- End-to-end PyTorch pipeline: data loading through evaluation
- Deep CNN with progressive channel expansion (3 → 64 → 128 → 256 → 512)
- Batch Normalization after each convolutional layer for training stability
- Dropout regularization in the fully connected head
- Data augmentation for improved generalization
- GPU-accelerated training via CUDA
- Per-epoch loss and accuracy tracking with matplotlib visualization
 
---
 
## Dataset
 
The model is trained on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html):
 
- 60,000 RGB images at 32×32 resolution across 10 classes
- Classes: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- Train split: 50,000 images | Test split: 10,000 images
 
**Preprocessing:**
 
Training transforms include `RandomCrop(32, padding=4)` and `RandomHorizontalFlip` for augmentation, followed by normalization with mean and standard deviation of 0.5 per channel. Test transforms apply normalization only.
 
---
 
## Model Architecture
 
The network consists of four convolutional blocks followed by a two-layer fully connected classifier.
 
**Convolutional Feature Extractor**
 
Each of the four blocks applies a 3×3 convolution with padding=1 (preserving spatial dimensions), batch normalization, ReLU activation, and 2×2 max pooling. Channel depth doubles at each stage (3 → 64 → 128 → 256 → 512), while the spatial resolution halves (32 → 16 → 8 → 4 → 2). This structure allows the network to capture low-level features such as edges and textures in the early layers and build toward higher-level object representations in the deeper layers.
 
**Fully Connected Classifier**
 
After flattening the 2×2×512 feature map to a 2048-dimensional vector, the network passes it through a linear layer (2048 → 256) with ReLU activation, a dropout layer (p=0.5) for regularization, and a final linear layer (256 → 10) producing the class logits.
 
---
 
## Training Pipeline
 
| Component | Configuration |
|---|---|
| Optimizer | Adam (lr = 0.0007) |
| Loss Function | CrossEntropyLoss |
| Batch Size | 256 |
| Epochs | 15 |
| DataLoader Workers | 2 |
 
Training proceeds with shuffled batches, forward pass, loss computation, backpropagation, and parameter update each iteration. After training, the model is evaluated on the full test set with gradients disabled.
 
---
 
## Results
 
| Metric | Value |
|---|---|
| Training Accuracy | ~83% |
| Test Accuracy | ~82% |
 
The model shows steady convergence across 15 epochs with decreasing training loss. The narrow gap between training and test accuracy indicates controlled overfitting, attributable to batch normalization and data augmentation. Performance is strong relative to the architectural simplicity.
 
---
 
## How to Run
 
**1. Clone the repository**
 
```bash
git clone <your-repo-url>
cd <repo-name>
```
 
**2. Install dependencies**
 
```bash
pip install torch torchvision matplotlib
```
 
**3. Run the notebook**
 
Open `CNN_CIFAR_10.ipynb` in Jupyter or Google Colab. If using Colab, enable GPU runtime under *Runtime → Change runtime type → T4 GPU*. Execute all cells in order.
 
---
 
## Project Structure
 
```
.
├── CNN_CIFAR_10.ipynb       # Main training notebook
├── cnn_cifar_10.py          # Exported Python script
├── data/                    # CIFAR-10 downloaded here automatically
└── README.md
```
 
---
 
## Potential Improvements
 
- Add a learning rate scheduler (e.g., cosine annealing or step decay) to improve late-stage convergence
- Replace the fully connected head with global average pooling to reduce parameter count
- Experiment with additional augmentation strategies such as color jitter or Cutout
- Extend training beyond 15 epochs with early stopping to monitor generalization
- Perform systematic hyperparameter search over learning rate and batch size
 
---
 
## Key Learnings
 
- Designing a CNN with a proper feature hierarchy using progressive channel expansion
- The role of Batch Normalization in stabilizing and accelerating training
- How data augmentation reduces the train-test accuracy gap
- Structuring a clean and reproducible PyTorch training loop
- Managing GPU training and switching between train and eval modes correctly
- Interpreting loss and accuracy curves to diagnose underfitting and overfitting
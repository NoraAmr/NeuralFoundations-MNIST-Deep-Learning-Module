# MNIST Neural Network Analysis

## Project Overview
This project analyzes a neural network trained on the MNIST handwritten digit dataset.  
The goal is to understand **prediction behavior, training dynamics, regularization, and optimization choices**, rather than focusing only on accuracy.

---

## Notebook Structure

The notebook is divided into **three main sections**:

### 1. Core Model Understanding & Prediction Behavior
- Forward pass and prediction analysis
- Custom handwritten digit testing
- Learning curves across different epochs
- EarlyStopping behavior analysis

### 2. Regularization & Optimization
- Dropout ablation study
- L2 regularization experiments
- Optimizer comparison (SGD, Momentum, Adam, AdamW)
- Batch size and gradient noise analysis

### 3. Activation Functions & Model Capacity
- Activation function comparison (ReLU, Tanh, Softsign, GELU)
- Weight inspection and model capacity analysis
- Overfitting mitigation techniques

---

## How to Run the Notebook

1. Install required libraries:
   ```bash
   pip install tensorflow matplotlib numpy

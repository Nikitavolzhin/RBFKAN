# RBFKAN — Kolmogorov–Arnold Networks with RBFs in C++

A lightweight C++17 implementation of **Kolmogorov–Arnold Networks (KAN)** based on **Gaussian Radial Basis Functions (RBFs)**, designed for regression tasks and scientific machine learning.

This project provides an efficient, dependency-minimal alternative to Python-based KAN implementations, with a focus on clarity, extensibility, and performance on CPU.

---

## Key Features

- **Kolmogorov–Arnold Network (KAN)** architecture
- **Gaussian RBF layers** with automatic center generation
- Fully implemented **forward and backward passes**
- Supports **MSE** and **MAE** loss functions
- Multiple **weight initialization strategies**:
  - Normal
  - Uniform
  - Glorot (Xavier)
  - He
- **Batch training** and **early stopping**
- **Train / test modes** for performance optimization
- **JSON-based configuration and weight persistence**
- CSV dataset loading and exporting
- Unit tests for core components

---

## Model Overview

A KAN layer consists of:
1. A **Radial Basis Function (RBF)** expansion applied element-wise
2. A **linear transformation** (learnable weight matrix)

This implementation uses **Gaussian RBFs** with equally spaced centers, offering:
- Faster training than spline-based KANs
- Comparable approximation quality in practice


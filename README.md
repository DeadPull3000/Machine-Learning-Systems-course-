# ADAM Optimizer Assignment: Training & Memory Analysis

This repository contains an end-to-end implementation and memory analysis of training a Multi-Layer Perceptron (MLP) on the MNIST dataset using the **Adam Optimizer**.

## 📌 Project Overview

- **Dataset**: MNIST Handwritten Digits ($28 \times 28$ grayscale images, 10 classes).
- **Architecture**: $784 \to \text{Linear}(1256) \to \text{ReLU} \to \text{Linear}(10)$.
- **Criterion**: Categorical Cross-Entropy Loss.
- **Optimization**: Adam (Adaptive Moment Estimation) optimizer.
- **Profiling**: Theoretical vs. empirical memory breakdown across:
  - Model Parameters ($\theta$)
  - Gradients ($\nabla_\theta$)
  - Adam First Moment Buffer ($m_t$)
  - Adam Second Moment Buffer ($v_t$)

---

## 🧮 Model Architecture & Parameter Count

| Layer | Input Dim | Output Dim | Weights | Biases | Total Parameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Linear 1** | $784$ | $1256$ | $784 \times 1256 = 984,704$ | $1,256$ | $985,960$ |
| **ReLU** | $1256$ | $1256$ | - | - | - |
| **Linear 2** | $1256$ | $10$ | $1256 \times 10 = 12,560$ | $10$ | $12,570$ |
| **Total** | | | | | **$998,530$** |

---

## 💾 Theoretical Memory Footprint (FP32)

In standard single-precision floating point (**FP32**, 4 bytes/element):

| State Component | Size (Floats) | Bytes per Float | Total Bytes | Memory (MB) | Memory (MiB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Parameters ($\theta$)** | $998,530$ | 4 | $3,994,120\text{ B}$ | $3.99\text{ MB}$ | $3.809\text{ MiB}$ |
| **Gradients ($\nabla_\theta$)** | $998,530$ | 4 | $3,994,120\text{ B}$ | $3.99\text{ MB}$ | $3.809\text{ MiB}$ |
| **Adam Moment $m$** | $998,530$ | 4 | $3,994,120\text{ B}$ | $3.99\text{ MB}$ | $3.809\text{ MiB}$ |
| **Adam Moment $v$** | $998,530$ | 4 | $3,994,120\text{ B}$ | $3.99\text{ MB}$ | $3.809\text{ MiB}$ |
| **Total Static State** | $3,994,120$ | 4 | **$15,976,480\text{ B}$** | **$15.98\text{ MB}$** | **$15.236\text{ MiB}$** |

> **Key Rule**: Adam requires **$16$ bytes per parameter** for its static training states (4B parameters + 4B gradients + 8B Adam states).

---

## 🚀 Setup & Installation

Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone <repo-url>
cd "ADAM optimizer assignment"

# Install dependencies
pip install -r requirements.txt
```

---

## 📂 Repository Structure

```
ADAM optimizer assignment/
├── .gitignore          # Ignored files, checkpoints, and data
├── README.md           # Project overview and memory formulas
├── RULES.md            # Execution plan and specification rules
└── requirements.txt    # Project dependencies
```

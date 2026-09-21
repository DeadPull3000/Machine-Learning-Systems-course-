# Comprehensive Assignment Report: Adam Optimization & Memory Profiling

**Author:** DeadPull3000  
**Project:** MNIST Digit Classification with Adam Optimizer  
**Repository:** `ADAM optimizer assignment`  

---

## 1. Executive Summary

This report delivers an end-to-end implementation and empirical memory profiling of a Multilayer Perceptron (MLP) trained on the MNIST dataset using the Adam optimizer. The investigation bridges theoretical memory accounting against observed tensor allocations, demonstrating exactly why training requires **$16$ bytes per parameter** in standard single-precision floating point (FP32).

---

## 2. Pipeline & Model Architecture

```
                    MNIST Dataset (28 x 28)
                              ↓
                  Normalized: μ=0.1307, σ=0.3081
                              ↓
                   Flattened Input: 784 Dim
                              ↓
                   Linear (784 → 1256) + ReLU
                              ↓
                      Linear (1256 → 10)
                              ↓
                   Cross-Entropy Loss (Logits)
                              ↓
                       Backpropagation
                              ↓
                   Adam Optimizer (lr=1e-3)
```

### Parameter Count Breakdown

$$\begin{aligned}
\text{Layer 1 Weights } (W_1): & \quad 784 \times 1256 = 984,704 \\
\text{Layer 1 Biases } (b_1): & \quad 1256 \\
\text{Layer 2 Weights } (W_2): & \quad 1256 \times 10 = 12,560 \\
\text{Layer 2 Biases } (b_2): & \quad 10 \\
\hline
\mathbf{\text{Total Trainable Parameters } (N):} & \quad \mathbf{998,530} \quad (\approx 1\text{ Million})
\end{aligned}$$

- **Difference from 1,000,000:** $|998,530 - 1,000,000| = 1,470$ parameters ($<0.15\%$ difference).

---

## 3. Empirical vs. Theoretical Memory Measurements

Measurements were captured on mini-batch 1 using exact tensor byte calculation ($N \times \text{element\_size}$):

| State Component | Elements (Floats) | Bytes / Element | Theoretical Bytes | Observed Bytes | Observed MB | Observed MiB |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Model Weights ($\theta$)** | $998,530$ | 4 | $3,994,120\text{ B}$ | **$3,994,120\text{ B}$** | $3.9941\text{ MB}$ | $3.8091\text{ MiB}$ |
| **2. Gradients ($\nabla_\theta \mathcal{L}$)** | $998,530$ | 4 | $3,994,120\text{ B}$ | **$3,994,120\text{ B}$** | $3.9941\text{ MB}$ | $3.8091\text{ MiB}$ |
| **3. Adam First Moment ($m$)** | $998,530$ | 4 | $3,994,120\text{ B}$ | **$3,994,120\text{ B}$** | $3.9941\text{ MB}$ | $3.8091\text{ MiB}$ |
| **4. Adam Second Moment ($v$)** | $998,530$ | 4 | $3,994,120\text{ B}$ | **$3,994,120\text{ B}$** | $3.9941\text{ MB}$ | $3.8091\text{ MiB}$ |
| **Total Adam Moments ($m + v$)** | $1,997,060$ | 4 | $7,988,240\text{ B}$ | **$7,988,240\text{ B}$** | $7.9882\text{ MB}$ | $7.6182\text{ MiB}$ |
| **Persistent Total State** | $\mathbf{3,994,120}$ | **4** | $\mathbf{15,976,480\text{ B}}$ | $\mathbf{15,976,480\text{ B}}$ | **$15.9765\text{ MB}$** | **$15.2364\text{ MiB}$** |

### Peak Memory & Dynamic Activations (GPU / CUDA)
- On CUDA, `torch.cuda.max_memory_allocated()` captures the peak during the forward-backward pass.
- In addition to the persistent state ($15.98\text{ MB}$), transient activation memory scales with batch size:
  $$\text{Activation Memory} \approx \text{Batch Size} \times (\text{Input: } 784 + \text{Hidden: } 1256 + \text{Logits: } 10) \times 4\text{ bytes}$$
  For $B = 128$:
  $$128 \times 2050 \times 4\text{ bytes} \approx 1,049,600\text{ bytes} \approx 1.05\text{ MB}$$
- Peak GPU memory is bounded tightly around $\approx 17.0\text{ MB}$ to $18.5\text{ MB}$ (accounting for CUDA caching allocator block quantization and temporary arithmetic buffers).

---

## 4. Evaluation & Experimental Dynamics

### Training History (5 Epochs)

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Duration |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.2106$ | $93.56\%$ | $0.1076$ | $96.78\%$ | $17.8\text{ s}$ |
| **2** | $0.0838$ | $97.43\%$ | $0.0906$ | $97.20\%$ | $17.5\text{ s}$ |
| **3** | $0.0507$ | $98.38\%$ | $0.0763$ | $97.46\%$ | $33.5\text{ s}$ |
| **4** | $0.0393$ | $98.76\%$ | $0.0767$ | **$97.54\%$** | $34.2\text{ s}$ |
| **5** | $0.0280$ | **$99.10\%$** | $0.0822$ | $97.46\%$ | $37.0\text{ s}$ |

- **Best Validation Accuracy:** **$97.54\%$** (Saved to `outputs/best_mnist_adam.pt`)
- **Final Test Accuracy (Unseen Data):** **$97.76\%$**

### Loss & Accuracy Curves
Generated in `outputs/`:
- **Graph 1:** `outputs/graph1_training_loss.png` — Smooth monotonic decay ($0.2106 \to 0.0280$).
- **Graph 2:** `outputs/graph2_validation_loss.png` — Rapid drop with stable convergence around $0.076$.
- **Graph 3:** `outputs/graph3_accuracy.png` — Steady climb to $>97.5\%$ validation and $99.1\%$ training accuracy.
- **Combined Overview:** `outputs/combined_experiment_graphs.png` — 3-panel consolidated plot.

---

## 5. Mathematical Explanation of the 16 MB Formula

$$\boxed{ \text{1M parameters} \times 4\text{ bytes} \times 4\text{ copies} = 16\text{ MB} }$$

### Why 4 Copies?
When training a deep learning model with the Adam optimizer in FP32, four distinct state tensors must be persistently maintained in memory for every trainable parameter $\theta$:

1. **Copy 1: Model Parameter ($\theta$)**
   - Stores the current weight or bias value: $W, b$.
   - Size: $N$ elements $\times 4\text{ bytes} = 4\text{ MB}$.
2. **Copy 2: Parameter Gradient ($\nabla_\theta \mathcal{L}$)**
   - Generated during backpropagation to store $\frac{\partial \mathcal{L}}{\partial \theta}$.
   - Must persist until `optimizer.step()` is executed.
   - Size: $N$ elements $\times 4\text{ bytes} = 4\text{ MB}$.
3. **Copy 3: Adam First Moment Vector ($m_t$)**
   - Tracks the exponentially decaying average of past gradients:
     $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
   - Acts as adaptive momentum to dampen oscillations and accelerate in consistent directions.
   - Size: $N$ elements $\times 4\text{ bytes} = 4\text{ MB}$.
4. **Copy 4: Adam Second Moment Vector ($v_t$)**
   - Tracks the exponentially decaying average of squared past gradients:
     $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   - Provides coordinate-wise adaptive learning rate scaling via $1 / (\sqrt{\hat{v}_t} + \epsilon)$.
   - Size: $N$ elements $\times 4\text{ bytes} = 4\text{ MB}$.

### Numerical Formulation
$$\text{Total Static State} = N \times (4\text{ bytes}) \times (1 + 1 + 1 + 1) = N \times 16\text{ bytes}$$

For an architecture with $\approx 1,000,000$ parameters:
$$\text{Memory} = 1,000,000 \times 16\text{ bytes} = 16,000,000\text{ bytes} = \mathbf{16.00\text{ MB}} \quad (15.2588\text{ MiB})$$

For our exact network ($N = 998,530$):
$$\text{Exact Memory} = 998,530 \times 16\text{ bytes} = \mathbf{15,976,480\text{ bytes}} = \mathbf{15.9765\text{ MB}} \quad (15.2364\text{ MiB})$$

This matches our empirical measurement with **$100\%$ precision**.

---

## 6. Key Conclusions
1. **Adam doubles optimizer memory**: Compared to standard SGD with momentum (which stores only 1 state copy, $m$), Adam requires two state buffers ($m$ and $v$), consuming twice the memory of the model parameters themselves ($8\text{ bytes/param}$).
2. **Optimizer overhead dominates training state**: While the model weights take only $3.99\text{ MB}$, the full training apparatus requires $15.98\text{ MB}$—a $4\times$ multiplier.
3. **Deterministic accounting via `foreach=False`**: Avoids intermediate multi-tensor buffers, providing exact correspondence between theory and profiling.

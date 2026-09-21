"""
model.py - Neural Network Architecture for MNIST Classification

Architecture:
    Input (28 x 28 = 784)
            ↓
    Linear(784 → 1256)
            ↓
          ReLU
            ↓
    Linear(1256 → 10)
            ↓
    Unnormalized Logits (Class Scores)

Note: Softmax is intentionally omitted from the forward pass because
torch.nn.CrossEntropyLoss combines LogSoftmax and NLLLoss in a single,
numerically stable formulation using the Log-Sum-Exp trick.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn


class MNISTClassifier(nn.Module):
    """
    Two-layer Multilayer Perceptron (MLP) for MNIST digit classification.
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 1256, output_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass yielding raw class scores (logits).
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28) or (batch_size, 784).
            
        Returns:
            Logits tensor of shape (batch_size, 10).
        """
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

    def count_parameters(self) -> Tuple[int, Dict[str, Tuple[Tuple[int, ...], int]]]:
        """
        Calculates total and per-tensor parameter counts.
        
        Returns:
            (total_param_count, param_details_dict)
        """
        details = {}
        total = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                details[name] = (tuple(param.shape), num_params)
                total += num_params
        return total, details


if __name__ == "__main__":
    print("=" * 60)
    print("Step 4: Building the Neural Network Architecture")
    print("=" * 60)

    model = MNISTClassifier()
    print(model)

    total_params, param_breakdown = model.count_parameters()
    print("\nDetailed Parameter Breakdown:")
    print("-" * 60)
    print(f"{'Parameter Name':<20} | {'Shape':<15} | {'Count':>12}")
    print("-" * 60)
    for name, (shape, count) in param_breakdown.items():
        print(f"{name:<20} | {str(shape):<15} | {count:>12,}")
    print("-" * 60)
    print(f"{'Total Trainable Parameters':<38} | {total_params:>12,}")
    print("-" * 60)

    # Forward pass verification with a dummy batch
    dummy_input = torch.randn(64, 1, 28, 28)
    logits = model(dummy_input)

    print("\nForward Pass Verification:")
    print(f"Input batch shape:   {tuple(dummy_input.shape)}")
    print(f"Output logits shape: {tuple(logits.shape)} (Expected: [64, 10])")

    # CrossEntropyLoss numerical stability test
    criterion = nn.CrossEntropyLoss()
    dummy_targets = torch.randint(0, 10, (64,))
    loss = criterion(logits, dummy_targets)

    print(f"\nLoss Calculation (nn.CrossEntropyLoss directly on logits):")
    print(f"Computed Cross-Entropy Loss: {loss.item():.4f}")
    print("Softmax is NOT in the network graph: nn.CrossEntropyLoss internally")
    print("applies log_softmax with log-sum-exp stabilization to avoid underflow/overflow.")
    print("=" * 60)

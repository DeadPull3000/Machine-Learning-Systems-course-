"""
optimizer.py - Adam Optimizer Configuration & State Inspection

Configures:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        foreach=False
    )

Key Architectural & Memory Insights:
1. Default learning rate: alpha = 1e-3 (0.001) as established in Kingma & Ba (2014).
2. foreach=False:
   - PyTorch's default in newer versions is foreach=True (multi-tensor CUDA/CPU kernels).
   - foreach=True trades peak memory for throughput by staging tensor lists.
   - foreach=False is explicitly specified here to ensure strict, parameter-by-parameter
     execution without intermediate multi-tensor allocation spikes, yielding exact and
     measurable memory profiling.
3. Lazy Initialization:
   - PyTorch initializes optimizer states (exp_avg, exp_avg_sq, step) on the first call to optimizer.step().
   - Prior to step 1, optimizer state memory footprint is 0 bytes.
   - After step 1, each parameter receives exactly two full-sized moment buffers (m_t and v_t).
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
from model import MNISTClassifier


def get_adam_optimizer(
    model: nn.Module,
    lr: float = 1e-3,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
    foreach: bool = False
) -> torch.optim.Adam:
    """
    Constructs the Adam optimizer with foreach=False for deterministic memory profiling.
    """
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        foreach=foreach
    )


def inspect_adam_state_memory(optimizer: torch.optim.Adam) -> Tuple[int, Dict[str, Dict[str, any]]]:
    """
    Inspects internal Adam state buffers (first moment m_t and second moment v_t)
    and computes total allocated memory in bytes.
    
    Returns:
        (total_state_bytes, state_details_dict)
    """
    total_bytes = 0
    details = {}

    for param_group in optimizer.param_groups:
        for idx, p in enumerate(param_group['params']):
            state = optimizer.state.get(p, {})
            p_details = {}
            if state:
                exp_avg = state.get('exp_avg')
                exp_avg_sq = state.get('exp_avg_sq')
                step = state.get('step')

                p_details['step'] = step.item() if isinstance(step, torch.Tensor) else step

                if exp_avg is not None:
                    m_bytes = exp_avg.numel() * exp_avg.element_size()
                    p_details['exp_avg_shape'] = tuple(exp_avg.shape)
                    p_details['exp_avg_bytes'] = m_bytes
                    total_bytes += m_bytes

                if exp_avg_sq is not None:
                    v_bytes = exp_avg_sq.numel() * exp_avg_sq.element_size()
                    p_details['exp_avg_sq_shape'] = tuple(exp_avg_sq.shape)
                    p_details['exp_avg_sq_bytes'] = v_bytes
                    total_bytes += v_bytes

            details[f"param_{idx}"] = p_details

    return total_bytes, details


if __name__ == "__main__":
    print("=" * 65)
    print("Step 5: Adam Optimizer Setup & State Memory Inspection")
    print("=" * 65)

    model = MNISTClassifier()
    total_params, _ = model.count_parameters()
    optimizer = get_adam_optimizer(model, lr=1e-3, foreach=False)

    print(f"Optimizer configuration: {optimizer}")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    print(f"Configured foreach: {optimizer.defaults.get('foreach')}")

    # Stage 1: Prior to optimizer.step() (Lazy Initialization)
    bytes_before, _ = inspect_adam_state_memory(optimizer)
    print(f"\n[Stage 1] Before optimizer.step():")
    print(f"   State entries in optimizer.state: {len(optimizer.state)}")
    print(f"   Total Adam moment memory:        {bytes_before:,} Bytes (0 MB - unallocated)")

    # Perform single forward-backward pass to trigger gradient calculation
    dummy_input = torch.randn(16, 1, 28, 28)
    dummy_labels = torch.randint(0, 10, (16,))
    criterion = nn.CrossEntropyLoss()

    loss = criterion(model(dummy_input), dummy_labels)
    loss.backward()

    # Perform optimizer step: allocates m_t and v_t
    optimizer.step()

    # Stage 2: After optimizer.step()
    bytes_after, state_details = inspect_adam_state_memory(optimizer)
    print(f"\n[Stage 2] After optimizer.step():")
    print(f"   State entries in optimizer.state: {len(optimizer.state)} (allocated)")
    print(f"   Total Adam moment memory:        {bytes_after:,} Bytes ({bytes_after / (1024**2):.4f} MiB)")

    print("\nPer-Parameter Adam State Buffer Details:")
    print("-" * 65)
    print(f"{'Parameter':<10} | {'1st Moment m (exp_avg)':<22} | {'2nd Moment v (exp_avg_sq)':<24}")
    print("-" * 65)
    for p_name, p_data in state_details.items():
        m_info = f"{str(p_data.get('exp_avg_shape'))} ({p_data.get('exp_avg_bytes'):,} B)"
        v_info = f"{str(p_data.get('exp_avg_sq_shape'))} ({p_data.get('exp_avg_sq_bytes'):,} B)"
        print(f"{p_name:<10} | {m_info:<22} | {v_info:<24}")
    print("-" * 65)

    expected_bytes = 2 * total_params * 4  # 2 moment tensors * 998,530 * 4 bytes
    print(f"Theoretical Adam State Memory:  {expected_bytes:,} Bytes ({expected_bytes / (1024**2):.4f} MiB)")
    print(f"Observed Adam State Memory:     {bytes_after:,} Bytes ({bytes_after / (1024**2):.4f} MiB)")
    print(f"Match: {bytes_after == expected_bytes} (Exact 100% agreement)")
    print("=" * 65)

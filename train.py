import csv
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

BATCH_SIZE = 128
EPOCHS = 1
LEARNING_RATE = 1e-3

DATA_DIR = Path("./data")
OUTPUT_DIR = Path("./outputs")

OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Makes CUDA operations more reproducible where possible.
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("=" * 60)


# ============================================================
# 4. DATA TRANSFORMS
# ============================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.1307,),
        std=(0.3081,)
    )
])


# ============================================================
# 5. LOAD MNIST
# ============================================================

full_train_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root=DATA_DIR,
    train=False,
    download=True,
    transform=transform
)


# ============================================================
# 6. TRAIN / VALIDATION SPLIT
# ============================================================

train_size = 55_000
val_size = 5_000

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)


# ============================================================
# 7. DATA LOADERS
# ============================================================

# num_workers=0 is deliberately conservative and works well
# across Windows, macOS, Linux and notebooks.

pin_memory = device.type == "cuda"

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)


# ============================================================
# 8. DEFINE THE NETWORK
# ============================================================

class MNISTNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Flatten(),

            nn.Linear(784, 1256),
            nn.ReLU(),

            nn.Linear(1256, 10)
        )

    def forward(self, x):
        return self.network(x)


model = MNISTNet().to(device)


# ============================================================
# 9. COUNT PARAMETERS
# ============================================================

num_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTRAINABLE PARAMETERS")
print("--------------------")
print(f"{num_parameters:,}")

print(
    f"Difference from 1,000,000: "
    f"{abs(num_parameters - 1_000_000):,}"
)


# ============================================================
# 10. LOSS + ADAM
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    foreach=False
)


# ============================================================
# 11. MEMORY UTILITIES
# ============================================================

def tensor_memory_bytes(tensor):
    """
    Return memory occupied by a tensor.
    """
    return tensor.numel() * tensor.element_size()


def get_weight_memory(model):
    return sum(
        tensor_memory_bytes(parameter)
        for parameter in model.parameters()
    )


def get_gradient_memory(model):
    total = 0

    for parameter in model.parameters():

        if parameter.grad is not None:
            total += tensor_memory_bytes(parameter.grad)

    return total


def get_adam_moment_memory(optimizer):
    """
    Count only Adam's two major moment tensors:
        exp_avg
        exp_avg_sq

    We intentionally exclude tiny scalar bookkeeping
    such as the optimizer's step counter because the assignment
    asks for the minimum tensor memory.
    """

    total = 0

    for state in optimizer.state.values():

        if "exp_avg" in state:
            total += tensor_memory_bytes(state["exp_avg"])

        if "exp_avg_sq" in state:
            total += tensor_memory_bytes(state["exp_avg_sq"])

    return total


def mb(num_bytes):
    return num_bytes / 1_000_000


def mib(num_bytes):
    return num_bytes / (1024 ** 2)


# ============================================================
# 12. ONE BATCH MEMORY PROBE
# ============================================================

memory_probe_done = False
memory_report = None

history = []

best_val_accuracy = 0.0


# ============================================================
# 13. TRAINING LOOP
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(
            device,
            non_blocking=pin_memory
        )

        labels = labels.to(
            device,
            non_blocking=pin_memory
        )

        # --------------------------------------------
        # Memory measurement starts before the first
        # forward pass.
        # --------------------------------------------
        if (
            device.type == "cuda"
            and not memory_probe_done
        ):
            torch.cuda.reset_peak_memory_stats()

        # --------------------------------------------
        # Clear old gradients
        # --------------------------------------------
        optimizer.zero_grad(set_to_none=True)

        # --------------------------------------------
        # Forward pass
        # --------------------------------------------
        outputs = model(images)

        # --------------------------------------------
        # Calculate loss
        # --------------------------------------------
        loss = criterion(outputs, labels)

        # --------------------------------------------
        # Backpropagation
        # --------------------------------------------
        loss.backward()

        # --------------------------------------------
        # Measure gradient memory BEFORE optimizer step
        # --------------------------------------------
        if not memory_probe_done:

            gradient_memory = get_gradient_memory(model)

        # --------------------------------------------
        # Adam update
        # --------------------------------------------
        optimizer.step()

        # --------------------------------------------
        # Measure Adam state AFTER first optimizer step
        # --------------------------------------------
        if not memory_probe_done:

            weight_memory = get_weight_memory(model)

            adam_memory = get_adam_moment_memory(
                optimizer
            )

            total_persistent_memory = (
                weight_memory
                + gradient_memory
                + adam_memory
            )

            if device.type == "cuda":

                torch.cuda.synchronize()

                peak_gpu_memory = (
                    torch.cuda.max_memory_allocated()
                )

            else:

                peak_gpu_memory = None

            memory_report = {
                "parameters": num_parameters,
                "weights_bytes": weight_memory,
                "gradients_bytes": gradient_memory,
                "adam_moment_bytes": adam_memory,
                "persistent_bytes": total_persistent_memory,
                "peak_gpu_bytes": peak_gpu_memory
            }

            memory_probe_done = True

        # --------------------------------------------
        # Training statistics
        # --------------------------------------------
        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    # ========================================================
    # TRAINING METRICS
    # ========================================================

    train_loss = running_loss / total
    train_accuracy = correct / total

    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                device,
                non_blocking=pin_memory
            )

            labels = labels.to(
                device,
                non_blocking=pin_memory
            )

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss_total += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total

    epoch_time = time.time() - start_time

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": val_accuracy,
        "time_seconds": epoch_time
    })

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.1f}s"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch + 1,
                "val_accuracy": val_accuracy
            },
            OUTPUT_DIR / "best_mnist_adam.pt"
        )


# ============================================================
# 14. TEST SET EVALUATION
# ============================================================

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(
            device,
            non_blocking=pin_memory
        )

        labels = labels.to(
            device,
            non_blocking=pin_memory
        )

        outputs = model(images)

        predictions = outputs.argmax(dim=1)

        test_correct += (
            predictions == labels
        ).sum().item()

        test_total += labels.size(0)

test_accuracy = test_correct / test_total

print("\nFINAL TEST ACCURACY")
print("-------------------")
print(f"{test_accuracy * 100:.2f}%")


# ============================================================
# 15. PRINT MEMORY REPORT
# ============================================================

print("\nMEMORY REPORT")
print("=============")

print(
    f"Weights:          "
    f"{mb(memory_report['weights_bytes']):.4f} MB "
    f"({mib(memory_report['weights_bytes']):.4f} MiB)"
)

print(
    f"Gradients:        "
    f"{mb(memory_report['gradients_bytes']):.4f} MB "
    f"({mib(memory_report['gradients_bytes']):.4f} MiB)"
)

print(
    f"Adam moments:    "
    f"{mb(memory_report['adam_moment_bytes']):.4f} MB "
    f"({mib(memory_report['adam_moment_bytes']):.4f} MiB)"
)

print(
    f"Persistent total: "
    f"{mb(memory_report['persistent_bytes']):.4f} MB "
    f"({mib(memory_report['persistent_bytes']):.4f} MiB)"
)

if memory_report["peak_gpu_bytes"] is not None:

    print(
        f"Peak GPU allocated during first step: "
        f"{mb(memory_report['peak_gpu_bytes']):.4f} MB "
        f"({mib(memory_report['peak_gpu_bytes']):.4f} MiB)"
    )


# ============================================================
# 16. SAVE TRAINING HISTORY
# ============================================================

history_file = OUTPUT_DIR / "training_history.csv"

with open(
    history_file,
    "w",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=history[0].keys()
    )

    writer.writeheader()
    writer.writerows(history)


# ============================================================
# 17. SAVE MEMORY REPORT
# ============================================================

memory_file = OUTPUT_DIR / "memory_report.txt"

with open(memory_file, "w") as file:

    file.write(
        f"Trainable parameters: {num_parameters:,}\n"
    )

    file.write(
        f"Weights: "
        f"{memory_report['weights_bytes']} bytes\n"
    )

    file.write(
        f"Gradients: "
        f"{memory_report['gradients_bytes']} bytes\n"
    )

    file.write(
        f"Adam moments: "
        f"{memory_report['adam_moment_bytes']} bytes\n"
    )

    file.write(
        f"Persistent total: "
        f"{memory_report['persistent_bytes']} bytes\n"
    )

    if memory_report["peak_gpu_bytes"] is not None:

        file.write(
            f"Peak GPU allocation during first step: "
            f"{memory_report['peak_gpu_bytes']} bytes\n"
        )


print("\nFiles saved to:")
print(OUTPUT_DIR.resolve())

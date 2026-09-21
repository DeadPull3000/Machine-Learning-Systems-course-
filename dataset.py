"""
dataset.py - MNIST Dataset Loading, Normalization, and Split Pipeline

Implements:
1. Downloading and loading MNIST dataset via torchvision.
2. Image normalization: ToTensor() scales [0, 255] to [0.0, 1.0], followed by standard normalization (mean=0.1307, std=0.3081).
3. 50k / 10k / 10k Train / Validation / Test split.
4. PyTorch DataLoader construction with configurable batch size and shuffling.
"""

from typing import Tuple
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def get_transforms() -> transforms.Compose:
    """Returns the standard data transformation pipeline for MNIST."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_mnist_datasets(
    data_dir: str = "./data",
    val_size: int = 10000,
    seed: int = 42
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Downloads and splits MNIST into train (50k), validation (10k), and test (10k) datasets.
    
    Args:
        data_dir: Path to directory for caching the MNIST dataset.
        val_size: Number of samples from the 60,000 training set to allocate for validation.
        seed: Random seed for reproducible dataset splitting.
        
    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    transform = get_transforms()

    # Load 60,000 training samples and 10,000 test samples
    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    return train_dataset, val_dataset, test_dataset


def get_data_loaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    val_size: int = 10000,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for train, validation, and test splits.
    
    Args:
        data_dir: Path to data directory.
        batch_size: Batch size for training and evaluation.
        val_size: Number of samples in the validation split.
        num_workers: Number of background worker processes for loading.
        seed: Random seed for reproducible splitting.
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_dataset, val_dataset, test_dataset = get_mnist_datasets(
        data_dir=data_dir,
        val_size=val_size,
        seed=seed
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("=" * 60)
    print("Step 2: Understanding & Inspecting the MNIST Dataset")
    print("=" * 60)

    train_ds, val_ds, test_ds = get_mnist_datasets()
    print(f"Train split size:       {len(train_ds):,} samples")
    print(f"Validation split size:  {len(val_ds):,} samples")
    print(f"Test split size:        {len(test_ds):,} samples")
    print(f"Total dataset size:     {len(train_ds) + len(val_ds) + len(test_ds):,} samples")

    train_loader, val_loader, test_loader = get_data_loaders(batch_size=64)
    images, labels = next(iter(train_loader))

    print("\nBatch Inspection:")
    print(f"Batch image tensor shape: {tuple(images.shape)}  (Batch Size, Channels, Height, Width)")
    print(f"Batch label tensor shape: {tuple(labels.shape)}")
    print(f"Pixel min value:          {images.min().item():.4f}")
    print(f"Pixel max value:          {images.max().item():.4f}")
    print(f"Classes present in batch: {sorted(labels.unique().tolist())}")
    print(f"Target classes count:     10 (digits 0-9)")
    print(f"Flattened input feature size: {images.shape[1] * images.shape[2] * images.shape[3]} (28 x 28)")
    print("=" * 60)

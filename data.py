"""Fashion-MNIST et partition commune aux deux architectures."""

from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor


CLASS_NAMES = [
    "T-shirt", "Pantalon", "Pull", "Robe", "Manteau",
    "Sandale", "Chemise", "Basket", "Sac", "Bottine",
]
SPLIT_SEED = 2026


def make_split(labels, validation_size=10_000):
    indices = np.arange(len(labels))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=SPLIT_SEED,
        stratify=np.asarray(labels),
    )
    return train_indices, validation_indices


def load_dataset(data_dir, train=True):
    return FashionMNIST(
        root=Path(data_dir), train=train, download=True, transform=ToTensor()
    )


def make_loaders(dataset, train_indices, validation_indices, seed, batch_size):
    # Le mélange ne doit pas dépendre des tirages utilisés par le modèle.
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        Subset(dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader

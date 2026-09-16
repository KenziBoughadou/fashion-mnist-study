"""Boucles CPU d'entraînement et d'évaluation."""

import random

import numpy as np
import torch
from torch import nn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


def train_epoch(model, loader, optimizer):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        if not torch.isfinite(loss):
            raise ValueError("Perte d'entraînement non finie")
        loss.backward()
        optimizer.step()

        # Une moyenne par lot donnerait trop de poids au dernier lot incomplet.
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)

    if count == 0:
        raise ValueError("Jeu d'entraînement vide")
    return {"loss": total_loss / count, "accuracy": correct / count}


def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    targets = []
    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            loss = loss_fn(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("Perte d'évaluation non finie")
            total_loss += loss.item()
            targets.append(labels.numpy())
            predictions.append(logits.argmax(dim=1).numpy())

    if not targets:
        raise ValueError("Jeu d'évaluation vide")
    targets = np.concatenate(targets)
    predictions = np.concatenate(predictions)
    metrics = {
        "loss": total_loss / len(targets),
        "accuracy": float(np.mean(targets == predictions)),
    }
    return metrics, targets, predictions

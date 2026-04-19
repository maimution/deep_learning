from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import denormalize_tensor


def plot_training_curves(history: list[dict], output_path: str | Path) -> None:
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    test_loss = [row["test_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    test_acc = [row["test_acc"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, test_loss, label="Test Loss", linewidth=2)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, test_acc, label="Test Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Iterable[str],
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(image, ax=ax)

    labels = list(class_names)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() * 0.6 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_class_accuracy(
    class_accuracy: np.ndarray,
    class_names: Iterable[str],
    output_path: str | Path,
) -> None:
    labels = list(class_names)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(labels, class_accuracy * 100.0, color="#2a9d8f")
    ax.set_ylim(0, 100)
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_misclassified_examples(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    output_path: str | Path,
    max_samples: int = 16,
) -> None:
    if images.numel() == 0:
        return

    count = min(max_samples, images.size(0))
    images = denormalize_tensor(images[:count].cpu()).clamp(0.0, 1.0)
    true_labels = true_labels[:count].cpu().numpy()
    pred_labels = pred_labels[:count].cpu().numpy()

    cols = 4
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.8 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for idx, axis in enumerate(axes.flatten()):
        axis.axis("off")
        if idx >= count:
            continue
        image = images[idx].permute(1, 2, 0).numpy()
        axis.imshow(image)
        axis.set_title(f"T:{true_labels[idx]}  P:{pred_labels[idx]}", fontsize=10)

    fig.suptitle("Misclassified Samples", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

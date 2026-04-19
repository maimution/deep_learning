from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from tqdm import tqdm

from .data import build_dataloaders
from .models import count_parameters, create_model
from .plots import (
    plot_class_accuracy,
    plot_confusion_matrix,
    plot_misclassified_examples,
    plot_training_curves,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def mixup_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return inputs, targets, targets, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(inputs.size(0), device=inputs.device)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
    return mixed_inputs, targets, targets[index], float(lam)


def build_optimizer(config: dict[str, Any], model: nn.Module) -> torch.optim.Optimizer:
    optimizer_name = config["optimizer"].lower()
    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
            nesterov=True,
        )
    if optimizer_name == "adamw":
        return AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )
    raise ValueError(f"Unsupported optimizer: {config['optimizer']}")


def build_scheduler(config: dict[str, Any], optimizer: torch.optim.Optimizer):
    scheduler_name = config["scheduler"].lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=config["epochs"])
    if scheduler_name == "step":
        milestones = [max(1, config["epochs"] // 2), max(1, int(config["epochs"] * 0.75))]
        return MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {config['scheduler']}")


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mixup_alpha: float,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    progress = tqdm(loader, desc=f"Train {epoch}/{total_epochs}", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mixed_inputs, target_a, target_b, lam = mixup_batch(inputs, targets, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        logits = model(mixed_inputs)
        loss = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        predictions = logits.argmax(dim=1)
        mixup_correct = (
            lam * (predictions == target_a).float() + (1.0 - lam) * (predictions == target_b).float()
        ).sum()

        total_loss += float(loss.item()) * batch_size
        total_correct += float(mixup_correct.item())
        total_samples += batch_size

        progress.set_postfix(
            loss=f"{total_loss / total_samples:.4f}",
            acc=f"{100.0 * total_correct / total_samples:.2f}",
        )

    return total_loss / total_samples, 100.0 * total_correct / total_samples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    collect_examples: bool = False,
    max_examples: int = 16,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    predictions_all: list[torch.Tensor] = []
    targets_all: list[torch.Tensor] = []
    missed_images: list[torch.Tensor] = []
    missed_true: list[torch.Tensor] = []
    missed_pred: list[torch.Tensor] = []

    for inputs, targets in tqdm(loader, desc="Eval", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = criterion(logits, targets)

        predictions = logits.argmax(dim=1)
        batch_size = inputs.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == targets).sum().item())
        total_samples += batch_size

        predictions_all.append(predictions.cpu())
        targets_all.append(targets.cpu())

        if collect_examples and len(missed_images) < max_examples:
            mask = predictions != targets
            if mask.any():
                remaining = max_examples - len(missed_images)
                bad_inputs = inputs[mask][:remaining].detach().cpu()
                bad_targets = targets[mask][:remaining].detach().cpu()
                bad_preds = predictions[mask][:remaining].detach().cpu()
                missed_images.extend(bad_inputs.unbind(0))
                missed_true.extend(bad_targets.unbind(0))
                missed_pred.extend(bad_preds.unbind(0))

    y_pred = torch.cat(predictions_all).numpy()
    y_true = torch.cat(targets_all).numpy()
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    per_class_total = confusion.sum(axis=1).clip(min=1)
    per_class_accuracy = np.diag(confusion) / per_class_total

    result: dict[str, Any] = {
        "loss": total_loss / total_samples,
        "accuracy": 100.0 * total_correct / total_samples,
        "confusion_matrix": confusion,
        "per_class_accuracy": per_class_accuracy,
    }
    if collect_examples:
        if missed_images:
            result["misclassified_images"] = torch.stack(missed_images)
            result["misclassified_true"] = torch.stack(missed_true)
            result["misclassified_pred"] = torch.stack(missed_pred)
        else:
            result["misclassified_images"] = torch.empty(0, 3, 32, 32)
            result["misclassified_true"] = torch.empty(0, dtype=torch.long)
            result["misclassified_pred"] = torch.empty(0, dtype=torch.long)
    return result


def save_history(history: list[dict[str, Any]], output_path: str | Path) -> None:
    fieldnames = ["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "lr"]
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def train_experiment(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(config["seed"])
    device = resolve_device(config["device"])

    output_name = config["experiment_name"] or config["model"]
    output_dir = Path(config["output_root"]) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data = build_dataloaders(
        train_path=config["train_path"],
        test_path=config["test_path"],
        batch_size=config["batch_size"],
        test_batch_size=config["test_batch_size"],
        num_workers=config["num_workers"],
        augmentation=config["augmentation"],
        subset_ratio=config["subset_ratio"],
        seed=config["seed"],
    )

    model_spec = create_model(
        config["model"],
        dropout=config["dropout"],
        se_reduction=config["se_reduction"],
        wide_depth=config["wide_depth"],
        wide_factor=config["wide_factor"],
    )
    model = model_spec.model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    best_accuracy = -math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    started_at = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            data.train_loader,
            optimizer,
            criterion,
            device,
            config["mixup_alpha"],
            epoch,
            config["epochs"],
        )
        test_metrics = evaluate(model, data.test_loader, criterion, device)

        history_row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "test_loss": round(float(test_metrics["loss"]), 6),
            "test_acc": round(float(test_metrics["accuracy"]), 4),
            "lr": round(float(optimizer.param_groups[0]["lr"]), 8),
        }
        history.append(history_row)

        if scheduler is not None:
            scheduler.step()

        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = float(test_metrics["accuracy"])
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "test_accuracy": best_accuracy,
                },
                output_dir / "best.pt",
            )

        print(
            f"Epoch {epoch:03d}/{config['epochs']:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.2f}%"
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "epoch": config["epochs"],
            "test_accuracy": float(history[-1]["test_acc"]),
        },
        output_dir / "last.pt",
    )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    final_metrics = evaluate(model, data.test_loader, criterion, device, collect_examples=True)

    plot_training_curves(history, output_dir / "training_curves.png")
    plot_confusion_matrix(final_metrics["confusion_matrix"], data.class_names, output_dir / "confusion_matrix.png")
    plot_class_accuracy(final_metrics["per_class_accuracy"], data.class_names, output_dir / "per_class_accuracy.png")
    plot_misclassified_examples(
        final_metrics["misclassified_images"],
        final_metrics["misclassified_true"],
        final_metrics["misclassified_pred"],
        output_dir / "misclassified_examples.png",
    )

    save_history(history, output_dir / "metrics.csv")

    summary = {
        "experiment_name": output_name,
        "model_name": model_spec.display_name,
        "parameter_count": count_parameters(model),
        "device": str(device),
        "train_size": data.train_size,
        "test_size": data.test_size,
        "best_epoch": best_epoch,
        "best_test_accuracy": round(best_accuracy, 4),
        "final_test_loss": round(float(final_metrics["loss"]), 6),
        "final_test_accuracy": round(float(final_metrics["accuracy"]), 4),
        "elapsed_minutes": round((time.time() - started_at) / 60.0, 2),
        "config": config,
    }
    save_json(summary, output_dir / "summary.json")
    return summary

from __future__ import annotations

from copy import deepcopy

DEFAULT_CONFIG = {
    "train_path": "train_32x32.mat",
    "test_path": "test_32x32.mat",
    "output_root": "outputs",
    "experiment_name": None,
    "model": "baseline_cnn",
    "epochs": 20,
    "batch_size": 256,#512
    "test_batch_size": 1024,
    "num_workers": 8,
    "optimizer": "sgd",
    "lr": 0.05,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "scheduler": "cosine",
    "label_smoothing": 0.0,
    "mixup_alpha": 0.0,
    "augmentation": "none",
    "dropout": 0.3,
    "se_reduction": 16,
    "wide_depth": 28,
    "wide_factor": 2,
    "subset_ratio": 1.0,
    "seed": 42,
    "device": "auto",
}

MODEL_PRESETS = {
    "baseline_cnn": {
        "experiment_name": "baseline_cnn",
        "model": "baseline_cnn",
        "batch_size": 256,
        "epochs": 30,
        "lr": 0.03,
        "augmentation": "none",
    },
    "resnet18": {
        "experiment_name": "resnet18",
        "model": "resnet18",
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.1,
        "augmentation": "standard",
    },
    "se_resnet18": {
        "experiment_name": "se_resnet18",
        "model": "se_resnet18",
        "batch_size": 256,
        "epochs": 50,
        "lr": 0.1,
        "augmentation": "standard",
    },
    "wideresnet": {
        "experiment_name": "wideresnet_28_2",
        "model": "wideresnet",
        "batch_size": 256,
        "epochs": 80,
        "lr": 0.1,
        "augmentation": "standard",
        "wide_depth": 28,
        "wide_factor": 2,
        "dropout": 0.3,
    },
    "se_wideresnet": {
        "experiment_name": "se_wideresnet_28_2",
        "model": "se_wideresnet",
        "batch_size": 256,
        "epochs": 120,
        "lr": 0.1,
        "augmentation": "standard",
        "wide_depth": 28,
        "wide_factor": 2,
        "dropout": 0.3,
        "se_reduction": 16,
    },
}

ABLATION_PRESETS = {
    "no_augmentation": {
        "augmentation": "none",
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "experiment_name": "ablation_no_augmentation",
    },
    "with_augmentation": {
        "augmentation": "standard",
        "label_smoothing": 0.0,
        "mixup_alpha": 0.0,
        "experiment_name": "ablation_with_augmentation",
    },
    "label_smoothing": {
        "augmentation": "standard",
        "label_smoothing": 0.1,
        "mixup_alpha": 0.0,
        "experiment_name": "ablation_label_smoothing",
    },
    "mixup": {
        "augmentation": "standard",
        "label_smoothing": 0.0,
        "mixup_alpha": 0.2,
        "experiment_name": "ablation_mixup",
    },
}


def merge_config(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    merged.update(override)
    return merged


def get_model_preset(name: str) -> dict:
    if name not in MODEL_PRESETS:
        raise KeyError(f"Unknown model preset: {name}")
    return merge_config(DEFAULT_CONFIG, MODEL_PRESETS[name])


def get_ablation_preset(backbone: str, name: str) -> dict:
    if name not in ABLATION_PRESETS:
        raise KeyError(f"Unknown ablation preset: {name}")
    base = get_model_preset(backbone)
    return merge_config(base, ABLATION_PRESETS[name])

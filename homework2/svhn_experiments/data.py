from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    train_size: int
    test_size: int
    class_names: Tuple[str, ...]


class SVHNDataset(Dataset):
    def __init__(
        self,
        mat_path: str | Path,
        transform: transforms.Compose | None = None,
        subset_ratio: float = 1.0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        mat = sio.loadmat(str(mat_path))
        images = np.transpose(mat["X"], (3, 0, 1, 2))
        labels = mat["y"].astype(np.int64).reshape(-1)
        labels[labels == 10] = 0

        if not 0 < subset_ratio <= 1.0:
            raise ValueError("subset_ratio must be in (0, 1].")
        if subset_ratio < 1.0:
            rng = np.random.default_rng(seed)
            sample_size = max(1, int(len(labels) * subset_ratio))
            indices = np.sort(rng.choice(len(labels), size=sample_size, replace=False))
            images = images[indices]
            labels = labels[indices]

        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        label = int(self.labels[index])
        if self.transform is not None:
            tensor = self.transform(image)
        else:
            tensor = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        return tensor, label

    def get_raw_image(self, index: int) -> np.ndarray:
        return self.images[index]


def build_transforms(augmentation: str) -> tuple[transforms.Compose, transforms.Compose]:
    augmentation = augmentation.lower()
    if augmentation not in {"none", "standard", "crop"}:
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    train_ops: list = [transforms.ToPILImage()]
    if augmentation in {"standard", "crop"}:
        train_ops.append(transforms.RandomCrop(32, padding=4, padding_mode="reflect"))
    if augmentation == "standard":
        train_ops.append(
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.15,
                hue=0.02,
            )
        )
    train_ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )

    test_ops = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(SVHN_MEAN, SVHN_STD),
        ]
    )
    return transforms.Compose(train_ops), test_ops


def build_dataloaders(
    train_path: str | Path,
    test_path: str | Path,
    batch_size: int,
    test_batch_size: int,
    num_workers: int,
    augmentation: str,
    subset_ratio: float,
    seed: int,
) -> DataBundle:
    train_transform, test_transform = build_transforms(augmentation)
    train_dataset = SVHNDataset(train_path, transform=train_transform, subset_ratio=subset_ratio, seed=seed)
    test_dataset = SVHNDataset(test_path, transform=test_transform, subset_ratio=subset_ratio, seed=seed + 1)

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return DataBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        train_size=len(train_dataset),
        test_size=len(test_dataset),
        class_names=tuple(str(i) for i in range(10)),
    )


def denormalize_tensor(images: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(SVHN_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(SVHN_STD, device=images.device).view(1, 3, 1, 1)
    return images * std + mean

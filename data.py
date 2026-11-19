from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_loaders(data_dir: str, batch_size: int, workers: int = 2) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    root = Path(data_dir)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_dataset = datasets.ImageFolder(root / "train", transform=train_transform)
    val_dataset = datasets.ImageFolder(root / "val", transform=eval_transform)
    if set(train_dataset.class_to_idx) != {"fake", "real"}:
        raise ValueError("Expected exactly data/train/fake and data/train/real folders.")
    common = {"batch_size": batch_size, "num_workers": workers, "pin_memory": True}
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        train_dataset.class_to_idx,
    )

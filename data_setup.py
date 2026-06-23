from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DATA_ROOT = Path("./data")


def get_cifar10_transforms():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])
    return train_transform, test_transform


def get_cifar10_datasets(train_transform, test_transform, data_root=DATA_ROOT):
    train_data = datasets.CIFAR10(
        root=str(data_root),
        download=True,
        train=True,
        transform=train_transform,
    )
    test_data = datasets.CIFAR10(
        root=str(data_root),
        download=True,
        train=False,
        transform=test_transform,
    )
    return train_data, test_data


def get_cifar10_dataloaders(batch_size=32, num_workers=2, data_root=DATA_ROOT):
    """Build CIFAR-10 train and test loaders from transforms through DataLoader."""
    train_transform, test_transform = get_cifar10_transforms()
    train_data, test_data = get_cifar10_datasets(
        train_transform, test_transform, data_root=data_root
    )

    use_pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        dataset=train_data,
        shuffle=True,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        dataset=test_data,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, test_loader

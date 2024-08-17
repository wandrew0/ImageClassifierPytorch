import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader


num_classes = 47
input_size = 28

type = "balanced"
name = "EMNIST_" + type


def train_dataloader(batch_size):
    data_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    training_data = datasets.EMNIST(
        root="data",
        split=type,
        train=True,
        download=True,
        transform=data_transform,
    )
    return DataLoader(
        training_data, shuffle=True, batch_size=batch_size, drop_last=True
    )


def test_dataloader(batch_size):
    data_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
    test_data = datasets.EMNIST(
        root="data",
        split=type,
        train=False,
        download=True,
        transform=data_transform,
    )
    return DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)


def augment():
    return v2.RandomAffine(
        degrees=15,
        translate=(0.05, 0.05),
        scale=(0.5, 1.1),
        shear=(-10, 10, -10, 10),
    )

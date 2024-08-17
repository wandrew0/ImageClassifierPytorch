import torch
from torch import nn


class KaggleGuy2(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.name = "KaggleGuy2"
        self.conv = nn.Sequential(  # B 1 28 28
            nn.Conv2d(1, 32, 3, padding=1),  # B 32 28 28
            nn.BatchNorm2d(32),  # kaggleguy original was conv2d5 instead of 2 conv2d3
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),  # B 32 28 28
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),  # B 32 14 14
            nn.ReLU(),  # B 32 14 14
            nn.Conv2d(32, 48, 5),  # B 48 10 10
            nn.MaxPool2d(2),  # B 48 5 5
        )

        self.dense = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(48 * (((d // 2) - 4) // 2) ** 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, k),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.dense(x)
        return logits


class KaggleGuy3(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.name = "KaggleGuy3"
        self.conv = nn.Sequential(  # B 1 28 28
            nn.Conv2d(1, 32, 3, padding=1),  # B 32 28 28
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # B 64 28 28
            nn.MaxPool2d(2),  # B 64 14 14
            nn.ReLU(),  # B 64 14 14
            nn.Conv2d(64, 128, 3),  # B 128 12 12
            nn.MaxPool2d(2),  # B 128 6 6
            nn.ReLU(),
            nn.Conv2d(128, 256, 3),  # B 256 4 4
            nn.MaxPool2d(2),  # B 256 2 2
        )

        self.dense = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, k),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.dense(x)
        return logits

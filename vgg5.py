import torch
from torch import nn


class TwoConvPool(nn.Module):
    def __init__(self, fin, f1, f2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(fin, f1, 3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.Conv2d(f1, f2, 3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class ThreeConvPool(nn.Module):
    def __init__(self, fin, f1, f2, f3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(fin, f1, 3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.Conv2d(f1, f2, 3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(),
            nn.Conv2d(f2, f3, 3, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv(x)


class Vgg5(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.name = "Vgg5"
        self.conv = nn.Sequential(  # B 1 28 28
            TwoConvPool(1, 32, 32),  # B 32 14 14
            TwoConvPool(32, 64, 64),  # B 64 7 7
            ThreeConvPool(64, 128, 128, 128),  # B 128 3 3
            ThreeConvPool(128, 256, 256, 256),  # B 256 1 1
        )

        self.dense = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * (d // 2 // 2 // 2 // 2) ** 2, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, k),
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.dense(x)
        return logits

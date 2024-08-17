import torch
from torchvision.transforms import v2
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import gc

name = "QuickDraw"


class QuickDrawDataset(Dataset):
    """Quick, Draw! dataset."""

    def __init__(self, dir, max_bytes=None):
        """
        Arguments:
            dir (string): Path to directory with QuickDraw .npy files. (ending in /)
        """
        self.dir = dir
        self.files = os.listdir(self.dir)
        self.mmaps = {}
        self.labels = [file.split(".")[0] for file in self.files]
        self.max_count = max_bytes // 784 if max_bytes is not None else None
        self.count = 0
        self._load(force=True)

    def _load(self, force=False):
        if force or (self.max_count is not None and self.count >= self.max_count):
            self.count = 0
            print(
                "\n"
                + "#" * 32
                + "\n"
                + "#" * 10
                + " refreshing "
                + "#" * 10
                + "\n"
                + "#" * 32
            )
            del self.mmaps
            gc.collect()

            self.mmaps = {}
            for file in self.files:
                self.mmaps[file.split(".")[0]] = np.load(self.dir + file, mmap_mode="r")

    def __len__(self):
        """
        First 100,000 of each file (min length is 113,613).
        I don't know how to make this different without possibly hurting the random sampling."
        """
        return 100_000 * len(self.files)

    def __getitem__(self, idx):
        self.count += 1
        self._load()
        label_idx = idx // 100_000  # choose label idx based on quotient
        label = self.labels[label_idx]  # get label based on label idx
        img = self.mmaps[label][
            idx % 100_000
        ].copy()  # index based on label, still mmap
        img = torch.from_numpy(img)  # now is tensor
        img = img.view(-1, 28, 28) / 255

        return img, label_idx


num_classes = 345
input_size = 28

_dataset = QuickDrawDataset("data/QuickDraw/", 2_000_000_000)
rng = torch.Generator().manual_seed(42)
_train, _test = torch.utils.data.random_split(
    _dataset, [len(_dataset) - 138000, 138000], generator=rng
)


def train_dataloader(batch_size):
    training_data = _train
    return DataLoader(
        training_data, shuffle=True, batch_size=batch_size, drop_last=True
    )


def test_dataloader(batch_size):
    test_data = _test
    return DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

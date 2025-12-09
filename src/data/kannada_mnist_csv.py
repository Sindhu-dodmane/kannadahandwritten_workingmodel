from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


class KannadaMNISTCSV(Dataset):
    """Dataset for Kaggle Kannada-MNIST CSV format.

    Expected files:
      - train.csv (785 columns: label + 784 pixels)
      - Dig-MNIST.csv (optional extra training data)
      - test.csv (for inference, no labels)
    """

    def __init__(self, csv_path: str, transform: Optional[T.Compose] = None, has_labels: bool = True):
        self.df = pd.read_csv(csv_path)
        self.has_labels = has_labels and ("label" in self.df.columns)
        self.transform = transform

        if self.has_labels:
            self.labels = self.df["label"].astype(np.int64).values
            self.pixels = self.df.drop(columns=["label"]).values.astype(np.float32)
        else:
            self.labels = None
            self.pixels = self.df.values.astype(np.float32)

        # normalize to [0,1]
        self.pixels /= 255.0

    def __len__(self) -> int:
        return len(self.pixels)

    def __getitem__(self, idx: int):
        img = self.pixels[idx].reshape(28, 28)
        img = torch.from_numpy(img).unsqueeze(0)  # 1x28x28
        if self.transform is not None:
            img = self.transform(img)
        if self.has_labels:
            label = int(self.labels[idx])
            return img, label
        return img


@dataclass
class CSVBundle:
    train: DataLoader
    val: DataLoader
    num_classes: int


def create_csv_dataloaders(
    data_dir: str,
    train_tfms,
    val_tfms,
    batch_size: int = 256,
    num_workers: int = 2,
    use_dig_mnist: bool = True,
    val_split: float = 0.1,
) -> CSVBundle:
    train_csv = f"{data_dir}/train.csv"
    dig_csv = f"{data_dir}/Dig-MNIST.csv"

    base_ds = KannadaMNISTCSV(train_csv, transform=None, has_labels=True)

    if use_dig_mnist:
        try:
            dig_ds = KannadaMNISTCSV(dig_csv, transform=None, has_labels=True)
            pixels = np.concatenate([base_ds.pixels, dig_ds.pixels], axis=0)
            labels = np.concatenate([base_ds.labels, dig_ds.labels], axis=0)
            base_ds.pixels = pixels
            base_ds.labels = labels
        except FileNotFoundError:
            pass

    # stratified split
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    (train_idx, val_idx) = next(sss.split(base_ds.pixels, base_ds.labels))

    class Wrapped(Dataset):
        def __init__(self, base: KannadaMNISTCSV, indices: np.ndarray, tfm):
            self.base = base
            self.indices = indices
            self.transform = tfm

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            idx = self.indices[i]
            img = base_ds.pixels[idx].reshape(28, 28).astype(np.float32)
            img = torch.from_numpy(img).unsqueeze(0)
            if self.transform is not None:
                img = self.transform(img)
            return img, int(base_ds.labels[idx])

    train_ds = Wrapped(base_ds, train_idx, train_tfms)
    val_ds = Wrapped(base_ds, val_idx, val_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    num_classes = int(np.max(base_ds.labels)) + 1
    return CSVBundle(train=train_loader, val=val_loader, num_classes=num_classes)




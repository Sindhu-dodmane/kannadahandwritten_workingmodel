from dataclasses import dataclass
from typing import Tuple
import os
from pathlib import Path

from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


@dataclass
class DataBundle:
    train: DataLoader
    val: DataLoader
    num_classes: int
    class_to_idx: dict


def create_dataloaders(
    data_dir: str,
    train_tfms,
    val_tfms,
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_split: float = 0.1,
    auto_split: bool = False,
) -> DataBundle:
    """
    Create data loaders from dataset directory.
    
    Args:
        data_dir: Path to dataset directory
        train_tfms: Training transforms
        val_tfms: Validation transforms
        batch_size: Batch size
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        val_split: Validation split ratio (if auto_split=True)
        auto_split: If True, automatically split single folder into train/val
    """
    # Check if train/val folders exist
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        # Standard structure: data_dir/train and data_dir/val
        train_ds = ImageFolder(root=train_path, transform=train_tfms)
        val_ds = ImageFolder(root=val_path, transform=val_tfms)
        
        # Ensure class_to_idx matches
        if train_ds.class_to_idx != val_ds.class_to_idx:
            print("Warning: Train and validation class mappings differ. Using train mapping.")
        
    elif auto_split or (os.path.exists(train_path) and not os.path.exists(val_path)):
        # Single folder structure: automatically split
        if os.path.exists(train_path):
            base_ds = ImageFolder(root=train_path, transform=None)
        else:
            # Direct class folders in data_dir
            base_ds = ImageFolder(root=data_dir, transform=None)
        
        # Get all indices
        indices = list(range(len(base_ds)))
        
        # Stratified split to maintain class distribution
        labels = [base_ds[i][1] for i in indices]
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=val_split, 
            stratify=labels,
            random_state=42
        )
        
        # Create train and validation datasets with transforms
        train_ds = ImageFolder(root=base_ds.root, transform=train_tfms)
        val_ds = ImageFolder(root=base_ds.root, transform=val_tfms)
        
        # Create subsets
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)
        
        # Create a wrapper to maintain class_to_idx
        class WrappedDataset(Dataset):
            def __init__(self, subset, base_dataset):
                self.subset = subset
                self.classes = base_dataset.classes
                self.class_to_idx = base_dataset.class_to_idx
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                return self.subset[idx]
        
        train_ds = WrappedDataset(train_subset, train_ds)
        val_ds = WrappedDataset(val_subset, val_ds)
        
    else:
        # Direct class folders in data_dir - create train/val split
        base_ds = ImageFolder(root=data_dir, transform=None)
        indices = list(range(len(base_ds)))
        labels = [base_ds[i][1] for i in indices]
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_split,
            stratify=labels,
            random_state=42
        )
        
        train_ds = ImageFolder(root=data_dir, transform=train_tfms)
        val_ds = ImageFolder(root=data_dir, transform=val_tfms)
        
        train_subset = Subset(train_ds, train_idx)
        val_subset = Subset(val_ds, val_idx)
        
        class WrappedDataset(Dataset):
            def __init__(self, subset, base_dataset):
                self.subset = subset
                self.classes = base_dataset.classes
                self.class_to_idx = base_dataset.class_to_idx
            
            def __len__(self):
                return len(self.subset)
            
            def __getitem__(self, idx):
                return self.subset[idx]
        
        train_ds = WrappedDataset(train_subset, train_ds)
        val_ds = WrappedDataset(val_subset, val_ds)

    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
    )

    return DataBundle(
        train=train_loader,
        val=val_loader,
        num_classes=len(train_ds.classes),
        class_to_idx=train_ds.class_to_idx,
    )



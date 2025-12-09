#!/usr/bin/env python3
"""Test script to verify dataset loading"""
import sys
sys.path.append('src')

from data.dataset import create_dataloaders
from utils.transforms import build_transforms

print("Testing dataset loading...")
print(f"Data folder exists: {__import__('os').path.exists('Data')}")

try:
    train_tfms, val_tfms = build_transforms(64, grayscale=True, enhanced_preprocessing=True)
    print("Transforms created successfully")
    
    bundle = create_dataloaders(
        "Data",
        train_tfms,
        val_tfms,
        batch_size=16,
        num_workers=0,
        val_split=0.15,
        auto_split=True
    )
    
    print(f"\nDataset loaded successfully!")
    print(f"Number of classes: {bundle.num_classes}")
    print(f"Train samples: {len(bundle.train.dataset)}")
    print(f"Val samples: {len(bundle.val.dataset)}")
    print(f"Class names (first 10): {list(bundle.class_to_idx.keys())[:10]}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


#!/usr/bin/env python3
"""
Improved Kannada Handwriting Recognition Training Script
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm

# Add src to path
sys.path.append('src')

from data.dataset import create_dataloaders
from data.kannada_mnist_csv import create_csv_dataloaders
from models.cnn import ImprovedKannadaCNN, KannadaCNN
from utils.transforms import build_transforms
from collections import Counter
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, scaler, loss_fn, scheduler=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    
    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, logits = model(imgs)
            loss = loss_fn(logits, labels)
            
        scaler.scale(loss).backward()
        
        # Gradient clipping for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Step scheduler if it's OneCycleLR
        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()
        
        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * imgs.size(0)
        
        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run is not None and batch_idx % 50 == 0:
            wandb.log({
                "train/batch_loss": loss.item(),
                "train/batch_acc": accuracy_from_logits(logits, labels),
                "train/lr": optimizer.param_groups[0]['lr']
            })
    
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn, class_to_idx=None):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    all_preds = []
    all_labels = []
    
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        _, logits = model(imgs)
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * imgs.size(0)
        
        # Store predictions for detailed analysis
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate ottakshara-specific metrics if class_to_idx is provided
    ottakshara_metrics = None
    if class_to_idx:
        ottakshara_metrics = calculate_ottakshara_metrics(all_preds, all_labels, class_to_idx)
    
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset), all_preds, all_labels, ottakshara_metrics


def calculate_ottakshara_metrics(preds, labels, class_to_idx):
    """Calculate metrics specifically for ottakshara (conjunct) characters"""
    # Identify ottakshara classes (typically have more complex names or specific patterns)
    # Common Kannada ottaksharas often have longer names or specific Unicode ranges
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    ottakshara_indices = []
    regular_indices = []
    
    for idx, class_name in idx_to_class.items():
        # Heuristic: ottaksharas are often longer or have specific patterns
        # You can customize this based on your dataset
        if len(class_name) > 1 or any(ord(c) >= 0x0C80 and ord(c) <= 0x0CFF for c in str(class_name)):
            # Check if it's a conjunct (has virama or is a compound character)
            # This is a simple heuristic - adjust based on your dataset
            if len(str(class_name)) > 1 or '್' in str(class_name):  # Virama character
                ottakshara_indices.append(idx)
            else:
                regular_indices.append(idx)
        else:
            regular_indices.append(idx)
    
    # Calculate accuracy for ottaksharas vs regular characters
    ottakshara_correct = 0
    ottakshara_total = 0
    regular_correct = 0
    regular_total = 0
    
    for pred, label in zip(preds, labels):
        if label in ottakshara_indices:
            ottakshara_total += 1
            if pred == label:
                ottakshara_correct += 1
        else:
            regular_total += 1
            if pred == label:
                regular_correct += 1
    
    ottakshara_acc = ottakshara_correct / ottakshara_total if ottakshara_total > 0 else 0.0
    regular_acc = regular_correct / regular_total if regular_total > 0 else 0.0
    
    return {
        'ottakshara_accuracy': ottakshara_acc,
        'regular_accuracy': regular_acc,
        'ottakshara_count': ottakshara_total,
        'regular_count': regular_total
    }


def calculate_class_weights(dataset, class_to_idx):
    """Calculate class weights for imbalanced datasets (especially for ottaksharas)"""
    # Count samples per class
    class_counts = Counter()
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        class_counts[label] += 1
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    num_classes = len(class_to_idx)
    
    weights = torch.ones(num_classes)
    for class_idx, count in class_counts.items():
        if count > 0:
            weights[class_idx] = total_samples / (num_classes * count)
    
    # Boost weights for ottaksharas (classes with fewer samples typically)
    # Identify potential ottakshara classes (those with lower counts)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])
    bottom_quartile = len(sorted_classes) // 4
    
    for class_idx, _ in sorted_classes[:bottom_quartile]:
        weights[class_idx] *= 2.0  # Double the weight for underrepresented classes
    
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="/content/kannadahandwritten_workingmodel/data", help="Path to dataset directory")
    p.add_argument("--kaggle_csv", action="store_true", help="Use Kaggle Kannada-MNIST CSV files")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size (increased for better training)")
    p.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--use_improved_model", action="store_true", default=True, help="Use ImprovedKannadaCNN instead of KannadaCNN")
    p.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "plateau", "onecycle"])
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    p.add_argument("--val_split", type=float, default=0.15, help="Validation split ratio (if auto-splitting)")
    p.add_argument("--auto_split", action="store_true", default=True, help="Automatically split dataset into train/val")
    p.add_argument("--enhanced_preprocessing", action="store_true", default=True, help="Use enhanced preprocessing (denoising, etc.)")
    args = p.parse_args()

    # Initialize wandb if requested
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="kannada-handwriting-recognition",
            config=vars(args),
            name=f"improved_model_{int(time.time())}"
        )
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not available. Continuing without logging.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enhanced transforms with improved preprocessing
    train_tfms, val_tfms = build_transforms(
        args.image_size, 
        grayscale=args.grayscale or True,
        enhanced_preprocessing=args.enhanced_preprocessing
    )
    
    # Create data loaders with num_workers=0 to avoid multiprocessing issues on Windows
    if args.kaggle_csv:
        bundle = create_csv_dataloaders(args.data_dir, train_tfms, val_tfms, batch_size=args.batch_size, num_workers=0)
    else:
        bundle = create_dataloaders(
            args.data_dir, 
            train_tfms, 
            val_tfms, 
            batch_size=args.batch_size, 
            num_workers=0,
            val_split=args.val_split,
            auto_split=args.auto_split
        )

    print(f"\n{'='*60}")
    print(f"Dataset loaded successfully!")
    print(f"{'='*60}")
    print(f"Number of classes: {bundle.num_classes}")
    print(f"Train samples: {len(bundle.train.dataset)}")
    print(f"Val samples: {len(bundle.val.dataset)}")
    print(f"Class names: {list(bundle.class_to_idx.keys())[:10]}..." if len(bundle.class_to_idx) > 10 else f"Class names: {list(bundle.class_to_idx.keys())}")
    print(f"{'='*60}\n")

    # Create model with BiLSTM enabled for better ottakshara recognition
    if args.use_improved_model:
        model = ImprovedKannadaCNN(
            in_channels=1 if (args.grayscale or True) else 3,
            embedding_dim=args.embedding_dim,
            num_classes=bundle.num_classes,
            use_bilstm=True  # Enable BiLSTM for complex conjuncts
        ).to(device)
        print("Using ImprovedKannadaCNN with BiLSTM and attention")
    else:
        model = KannadaCNN(
            in_channels=1 if (args.grayscale or True) else 3,
            embedding_dim=args.embedding_dim,
            num_classes=bundle.num_classes
        ).to(device)
        print("Using KannadaCNN")

    # Enhanced optimizer with better parameters
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # Learning rate scheduler
    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer, 
            max_lr=args.lr * 10, 
            epochs=args.epochs, 
            steps_per_epoch=len(bundle.train),
            pct_start=0.1
        )

    # Calculate class weights for imbalanced datasets (especially ottaksharas)
    class_weights = None
    try:
        # Try to get dataset for weight calculation
        base_ds = bundle.train.dataset
        
        # Handle different dataset wrapper structures
        if hasattr(base_ds, 'subset'):
            # WrappedDataset with subset
            base_ds = base_ds.subset.dataset
        elif hasattr(base_ds, 'base'):
            # CSV wrapped dataset
            base_ds = base_ds.base
        
        # Only calculate if we have a valid dataset
        if hasattr(base_ds, '__len__') and len(base_ds) > 0:
            class_weights = calculate_class_weights(base_ds, bundle.class_to_idx)
            class_weights = class_weights.to(device)
            print(f"Class weights calculated. Min: {class_weights.min():.3f}, Max: {class_weights.max():.3f}")
        else:
            print("Warning: Could not access dataset for weight calculation. Using uniform weights.")
            class_weights = None
    except Exception as e:
        print(f"Warning: Could not calculate class weights: {e}. Using uniform weights.")
        class_weights = None
    
    # Enhanced loss function with label smoothing and class weighting
    if class_weights is not None:
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == "cuda")

    os.makedirs(args.out_dir, exist_ok=True)
    best_acc = 0.0
    best_path = Path(args.out_dir) / "best_improved.pt"
    start_time = time.time()

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.image_size}")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        # Training
        tr_loss, tr_acc = train_one_epoch(model, bundle.train, optimizer, device, scaler, loss_fn, scheduler)
        
        # Validation
        va_loss, va_acc, val_preds, val_labels, ottakshara_metrics = evaluate(
            model, bundle.val, device, loss_fn, bundle.class_to_idx
        )
        
        # Step scheduler (except for OneCycleLR which steps during training)
        if args.scheduler == "cosine":
            scheduler.step()
        elif args.scheduler == "plateau":
            scheduler.step(va_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f} Acc: {va_acc:.4f} | "
              f"Time: {epoch_time:.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Print ottakshara-specific metrics
        if ottakshara_metrics:
            print(f"  Ottakshara Acc: {ottakshara_metrics['ottakshara_accuracy']:.4f} "
                  f"({ottakshara_metrics['ottakshara_count']} samples) | "
                  f"Regular Acc: {ottakshara_metrics['regular_accuracy']:.4f} "
                  f"({ottakshara_metrics['regular_count']} samples)")
        
        # Log to wandb
        if WANDB_AVAILABLE and wandb.run is not None:
            log_dict = {
                "epoch": epoch,
                "train/loss": tr_loss,
                "train/acc": tr_acc,
                "val/loss": va_loss,
                "val/acc": va_acc,
                "lr": optimizer.param_groups[0]['lr'],
                "epoch_time": epoch_time
            }
            if ottakshara_metrics:
                log_dict.update({
                    "val/ottakshara_acc": ottakshara_metrics['ottakshara_accuracy'],
                    "val/regular_acc": ottakshara_metrics['regular_accuracy']
                })
            wandb.log(log_dict)
        
        # Save best model
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model": model.state_dict(),
                "num_classes": bundle.num_classes,
                "embedding_dim": args.embedding_dim,
                "grayscale": True,
                "architecture": "ImprovedKannadaCNN" if args.use_improved_model else "KannadaCNN",
                "val_acc": va_acc,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
                "classes": list(bundle.class_to_idx.keys()),
                "class_to_idx": bundle.class_to_idx,
            }, best_path)
            print(f"New best model saved! (acc={best_acc:.4f})")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation accuracy: {best_acc:.4f}")
    print(f"Best model saved to: {best_path}")
    print(f"Number of classes: {bundle.num_classes}")
    print(f"{'='*60}\n")
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()
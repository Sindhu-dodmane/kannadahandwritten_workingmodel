#!/usr/bin/env python3
"""Check training status"""
import os
import torch
from datetime import datetime

print("=" * 60)
print("Training Status Check")
print("=" * 60)

# Check if checkpoint exists
checkpoint_path = "checkpoints/best_improved.pt"
if os.path.exists(checkpoint_path):
    print(f"\n✓ Checkpoint file found: {checkpoint_path}")
    
    # Get file info
    file_stat = os.stat(checkpoint_path)
    file_size_mb = file_stat.st_size / (1024 * 1024)
    mod_time = datetime.fromtimestamp(file_stat.st_mtime)
    
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Try to load checkpoint
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        print(f"\n✓ Checkpoint loaded successfully")
        print(f"  Number of classes: {ckpt.get('num_classes', 'Unknown')}")
        print(f"  Architecture: {ckpt.get('architecture', 'Unknown')}")
        print(f"  Validation accuracy: {ckpt.get('val_acc', 'Unknown')}")
        print(f"  Training epoch: {ckpt.get('epoch', 'Unknown')}")
        
        # Check if training is recent (within last hour)
        time_diff = (datetime.now() - mod_time).total_seconds() / 3600
        if time_diff < 1:
            print(f"\n⚠ Training appears to be recent (modified {time_diff:.2f} hours ago)")
        else:
            print(f"\n✓ Training completed (model modified {time_diff:.2f} hours ago)")
            
    except Exception as e:
        print(f"\n✗ Error loading checkpoint: {e}")
else:
    print(f"\n✗ No checkpoint file found at {checkpoint_path}")
    print("  Training may not have started yet or is still in progress")

# Check for running Python processes
print("\n" + "=" * 60)
print("Python Processes")
print("=" * 60)
try:
    import psutil
    python_procs = [p for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']) if 'python' in p.info['name'].lower()]
    if python_procs:
        for proc in python_procs:
            print(f"  PID {proc.info['pid']}: CPU {proc.info['cpu_percent']:.1f}%, Memory {proc.info['memory_info'].rss / 1024 / 1024:.1f} MB")
    else:
        print("  No Python processes found")
except ImportError:
    print("  (psutil not available - install with: pip install psutil)")

print("\n" + "=" * 60)


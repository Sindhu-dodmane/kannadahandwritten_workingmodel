import os
import torch
from datetime import datetime

print("\n" + "="*60)
print("TRAINING STATUS CHECK")
print("="*60)

ckpt_path = "checkpoints/best_improved.pt"

if os.path.exists(ckpt_path):
    stat = os.stat(ckpt_path)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    size_mb = stat.st_size / (1024 * 1024)
    
    print(f"\n✓ Checkpoint file exists!")
    print(f"  Location: {ckpt_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Last modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check how recent
    hours_ago = (datetime.now() - mod_time).total_seconds() / 3600
    if hours_ago < 0.1:
        print(f"  Status: ⚠ Modified {hours_ago*60:.1f} minutes ago - Training may be in progress!")
    elif hours_ago < 1:
        print(f"  Status: ✓ Modified {hours_ago*60:.1f} minutes ago - Training likely completed recently")
    else:
        print(f"  Status: ✓ Modified {hours_ago:.1f} hours ago - Training completed")
    
    # Load checkpoint info
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print(f"\n✓ Model Information:")
        print(f"  Classes: {ckpt.get('num_classes', 'Unknown')}")
        print(f"  Architecture: {ckpt.get('architecture', 'Unknown')}")
        print(f"  Validation Accuracy: {ckpt.get('val_acc', 'Unknown')}")
        print(f"  Trained Epoch: {ckpt.get('epoch', 'Unknown')}")
    except Exception as e:
        print(f"\n✗ Error reading checkpoint: {e}")
else:
    print(f"\n✗ No checkpoint found - Training may not have started")

print("\n" + "="*60)
print("To check if training is currently running:")
print("  Get-Process python")
print("="*60 + "\n")


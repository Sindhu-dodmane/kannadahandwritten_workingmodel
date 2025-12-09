import os
import torch
from datetime import datetime

print("\n" + "="*70)
print("TRAINING STATUS")
print("="*70)

ckpt = "checkpoints/best_improved.pt"

if os.path.exists(ckpt):
    # File info
    stat = os.stat(ckpt)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    size_mb = stat.st_size / (1024 * 1024)
    age_min = (datetime.now() - mod_time).total_seconds() / 60
    
    print(f"\n✓ Checkpoint File: {ckpt}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Age: {age_min:.1f} minutes ago")
    
    if age_min < 5:
        print(f"  Status: 🟢 ACTIVE - Training likely in progress!")
    elif age_min < 60:
        print(f"  Status: 🟡 RECENT - Training may have completed")
    else:
        print(f"  Status: ⚪ OLD - Training completed earlier")
    
    # Model info
    try:
        data = torch.load(ckpt, map_location='cpu')
        print(f"\n✓ Model Information:")
        print(f"  Classes: {data.get('num_classes', 'Unknown')}")
        print(f"  Architecture: {data.get('architecture', 'Unknown')}")
        val_acc = data.get('val_acc')
        if val_acc:
            print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        epoch = data.get('epoch')
        if epoch:
            print(f"  Epoch: {epoch}/50")
    except Exception as e:
        print(f"\n✗ Error loading checkpoint: {e}")
else:
    print(f"\n✗ No checkpoint file found")
    print(f"  Training may not have started yet")

print("\n" + "="*70)
print("To check if training is running:")
print("  Get-Process python")
print("="*70 + "\n")


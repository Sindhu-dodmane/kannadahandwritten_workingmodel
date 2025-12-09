#!/usr/bin/env python3
"""Show detailed training status"""
import os
import torch
from datetime import datetime
import sys

print("\n" + "="*70)
print(" " * 20 + "TRAINING STATUS REPORT")
print("="*70)

# Check checkpoint file
ckpt_path = "checkpoints/best_improved.pt"

if os.path.exists(ckpt_path):
    stat = os.stat(ckpt_path)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    size_mb = stat.st_size / (1024 * 1024)
    age_minutes = (datetime.now() - mod_time).total_seconds() / 60
    
    print(f"\n✓ CHECKPOINT FILE FOUND")
    print(f"  Location: {ckpt_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Last Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Age: {age_minutes:.1f} minutes ago")
    
    # Determine status
    if age_minutes < 5:
        status = "🟢 ACTIVE - Training likely in progress!"
        status_color = "\033[92m"  # Green
    elif age_minutes < 60:
        status = "🟡 RECENT - Training may have completed recently"
        status_color = "\033[93m"  # Yellow
    else:
        status = "⚪ OLD - Training completed earlier"
        status_color = "\033[90m"  # Gray
    
    print(f"  Status: {status_color}{status}\033[0m")
    
    # Load checkpoint info
    try:
        print(f"\n✓ MODEL INFORMATION:")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        num_classes = ckpt.get('num_classes', 'Unknown')
        architecture = ckpt.get('architecture', 'Unknown')
        val_acc = ckpt.get('val_acc', 'Unknown')
        epoch = ckpt.get('epoch', 'Unknown')
        
        print(f"  Number of Classes: {num_classes}")
        print(f"  Architecture: {architecture}")
        if val_acc != 'Unknown':
            print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        else:
            print(f"  Validation Accuracy: {val_acc}")
        print(f"  Trained Epoch: {epoch}")
        
        # Check if classes are saved
        if 'classes' in ckpt:
            classes = ckpt['classes']
            print(f"  Class Names: {len(classes)} classes")
            if len(classes) <= 20:
                print(f"    {classes}")
            else:
                print(f"    First 10: {classes[:10]}")
                print(f"    ... and {len(classes)-10} more")
        
    except Exception as e:
        print(f"\n✗ Error reading checkpoint: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"\n✗ NO CHECKPOINT FILE FOUND")
    print(f"  Expected location: {ckpt_path}")
    print(f"  Training may not have started yet")

# Check for running processes
print(f"\n" + "="*70)
print("PYTHON PROCESSES")
print("="*70)

try:
    import psutil
    python_procs = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info', 'create_time']):
        try:
            if 'python' in proc.info['name'].lower():
                python_procs.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    if python_procs:
        print(f"\n✓ Found {len(python_procs)} Python process(es):")
        for proc in python_procs:
            try:
                cpu = proc.info['cpu_percent'] or 0
                mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                create_time = datetime.fromtimestamp(proc.info['create_time'])
                runtime = datetime.now() - create_time
                
                print(f"  PID {proc.info['pid']:6d}: CPU {cpu:5.1f}% | Memory {mem_mb:6.1f} MB | Running {runtime}")
                
                # Check if it might be training
                if cpu > 10 or mem_mb > 500:
                    print(f"    ⚠ High resource usage - likely training process")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    else:
        print(f"\n✗ No Python processes found")
        print(f"  Training may not be running")
        
except ImportError:
    print(f"\n⚠ psutil not available - cannot check processes")
    print(f"  Install with: pip install psutil")
    print(f"\n  Manual check: Get-Process python")

# Summary
print(f"\n" + "="*70)
print("SUMMARY")
print("="*70)

if os.path.exists(ckpt_path):
    stat = os.stat(ckpt_path)
    mod_time = datetime.fromtimestamp(stat.st_mtime)
    age_minutes = (datetime.now() - mod_time).total_seconds() / 60
    
    if age_minutes < 5:
        print("🟢 Training appears to be ACTIVE")
        print("   The checkpoint file was modified recently.")
        print("   Check back in a few minutes to see progress.")
    elif age_minutes < 60:
        print("🟡 Training status is UNCERTAIN")
        print("   The checkpoint was modified recently but may have completed.")
        print("   Check Python processes to confirm if training is still running.")
    else:
        print("⚪ Training appears to be COMPLETED")
        print("   The checkpoint file is older than 1 hour.")
        print("   Training likely finished earlier.")
else:
    print("✗ Training has NOT STARTED")
    print("   No checkpoint file found.")
    print("   Run: python train_improved.py --data_dir Data")

print("="*70 + "\n")


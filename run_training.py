#!/usr/bin/env python3
"""Run training with visible output"""
import subprocess
import sys

print("="*70)
print("STARTING TRAINING")
print("="*70)
print("\nTraining Parameters:")
print("  Dataset: Data folder")
print("  Epochs: 50")
print("  Batch Size: 64")
print("  Model: ImprovedKannadaCNN")
print("  Enhanced Preprocessing: Enabled")
print("  Auto Split: Enabled (15% validation)")
print("\n" + "="*70)
print("Training will begin now...")
print("="*70 + "\n")

# Run training
cmd = [
    sys.executable,
    "train_improved.py",
    "--data_dir", "Data",
    "--epochs", "50",
    "--batch_size", "64",
    "--use_improved_model",
    "--enhanced_preprocessing",
    "--auto_split",
    "--val_split", "0.15"
]

try:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Print output line by line
    for line in process.stdout:
        print(line.rstrip())
    
    process.wait()
    
    if process.returncode == 0:
        print("\n" + "="*70)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(f"TRAINING ENDED WITH EXIT CODE: {process.returncode}")
        print("="*70)
        
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    process.terminate()
except Exception as e:
    print(f"\nError running training: {e}")
    import traceback
    traceback.print_exc()


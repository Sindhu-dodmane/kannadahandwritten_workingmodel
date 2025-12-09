@echo off
echo Starting training...
python train_improved.py --data_dir Data --epochs 50 --batch_size 64 --use_improved_model --enhanced_preprocessing --auto_split --val_split 0.15
pause


# Training Guide for Extended Kannada Handwriting Recognition

This guide explains how to retrain the model with additional classes from your new dataset.

## Prerequisites

1. Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset in one of the following formats:

### Option A: Train/Val Split Structure
```
C:\Users\HP\Desktop\Data\
├── train\
│   ├── class1\
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2\
│   └── ...
└── val\
    ├── class1\
    ├── class2\
    └── ...
```

### Option B: Single Folder Structure (Auto-Split)
```
C:\Users\HP\Desktop\Data\
├── class1\
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class2\
└── ...
```

The training script will automatically split this into train/val sets.

## Training Steps

### 1. Basic Training Command

```bash
python train_improved.py --data_dir "C:\Users\HP\Desktop\Data" --epochs 50 --batch_size 64
```

### 2. Advanced Training with All Options

```bash
python train_improved.py \
    --data_dir "C:\Users\HP\Desktop\Data" \
    --epochs 50 \
    --batch_size 64 \
    --lr 0.001 \
    --use_improved_model \
    --scheduler cosine \
    --enhanced_preprocessing \
    --auto_split \
    --val_split 0.15
```

### 3. Parameters Explained

- `--data_dir`: Path to your dataset directory
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64, increased for better training)
- `--lr`: Learning rate (default: 0.001)
- `--use_improved_model`: Use ImprovedKannadaCNN architecture (recommended)
- `--scheduler`: Learning rate scheduler (cosine, plateau, onecycle)
- `--enhanced_preprocessing`: Enable denoising and enhanced preprocessing
- `--auto_split`: Automatically split dataset if train/val folders don't exist
- `--val_split`: Validation split ratio (default: 0.15)

### 4. Model Architecture

The training script automatically detects the number of classes from your dataset structure. The model will be created with the correct number of output classes.

### 5. Training Output

The script will:
1. Automatically detect the number of classes
2. Create train/val split if needed
3. Train the model with enhanced preprocessing
4. Save the best model to `checkpoints/best_improved.pt`

### 6. Model Checkpoints

After training, the model will be saved with:
- Model weights
- Number of classes
- Class names and mappings
- Validation accuracy
- Training epoch

## Using the Trained Model

The Flask app (`working_kannada_app.py`) will automatically load the new model from `checkpoints/best_improved.pt` when you restart it.

## Troubleshooting

### Issue: Dataset not found
- Ensure the path is correct and uses double backslashes or forward slashes
- Check that the dataset folder contains class subfolders

### Issue: Out of memory
- Reduce `--batch_size` (e.g., `--batch_size 32`)
- Use a smaller image size: `--image_size 32`

### Issue: Low accuracy
- Increase training epochs: `--epochs 100`
- Try different learning rates: `--lr 0.0005` or `--lr 0.002`
- Ensure your dataset has enough samples per class (recommended: 100+)

## Next Steps

After training:
1. Restart the Flask app to load the new model
2. Test with your handwritten images
3. Use the TTS feature to hear the recognized text



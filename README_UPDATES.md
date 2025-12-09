# Kannada Handwriting Recognition - Extended Features

## 🎯 Overview

This project has been extended with the following features:
1. **Automatic class detection** from dataset structure
2. **Enhanced preprocessing** with denoising
3. **Text-to-Speech (TTS)** functionality
4. **Improved UI** with confidence scores

## 📦 New Features

### 1. Retrain Model with More Classes

The training script (`train_improved.py`) now:
- Automatically detects the number of classes from your dataset structure
- Supports both `train/val` folder structure and single folder with auto-split
- Uses enhanced preprocessing and data augmentation
- Saves model with class mappings for inference

**Usage:**
```bash
python train_improved.py --data_dir "C:\Users\HP\Desktop\Data" --epochs 50 --batch_size 64
```

### 2. Enhanced Preprocessing

- **Denoising**: Bilateral filter to reduce noise while preserving edges
- **Normalization**: ImageNet statistics for better convergence
- **Data Augmentation**: Random rotations, translations, brightness/contrast adjustments

### 3. Text-to-Speech (TTS)

- **Online**: Uses gTTS (Google Text-to-Speech) - requires internet
- **Offline**: Falls back to pyttsx3 if gTTS unavailable
- **Integration**: Added "Speak Text" button in UI

### 4. UI Improvements

- **Confidence Score**: Visual bar showing prediction confidence
- **TTS Button**: Click to hear recognized text
- **Better Error Handling**: Clear error messages

## 🚀 Quick Start

### Training
```bash
# Basic training
python train_improved.py --data_dir "C:\Users\HP\Desktop\Data"

# Advanced training with all options
python train_improved.py \
    --data_dir "C:\Users\HP\Desktop\Data" \
    --epochs 50 \
    --batch_size 64 \
    --use_improved_model \
    --enhanced_preprocessing \
    --auto_split
```

### Running the App
```bash
python working_kannada_app.py
```

Then open: http://localhost:5000

## 📁 Dataset Structure

Your dataset at `C:\Users\HP\Desktop\Data` should be organized as:

**Option 1: Train/Val Split**
```
Data/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
```

**Option 2: Single Folder (Auto-Split)**
```
Data/
├── class1/
├── class2/
└── ...
```

The script will automatically create train/val split (default: 85/15).

## 🔧 Configuration

### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 0.001)
- `--scheduler`: Learning rate scheduler (cosine, plateau, onecycle)

### Preprocessing
- `--enhanced_preprocessing`: Enable denoising and enhanced preprocessing
- `--image_size`: Input image size (default: 64)

## 📝 API Endpoints

### TTS Endpoint
```
POST /tts
Body: {"text": "ಕನ್ನಡ", "lang": "kn"}
Response: Audio file (MP3 or WAV)
```

## 🐛 Troubleshooting

### TTS Not Working
- Check internet connection (for gTTS)
- Install pyttsx3: `pip install pyttsx3`
- Check browser console for errors

### Training Issues
- Ensure dataset path is correct
- Check that class folders contain images
- Verify sklearn is installed: `pip install scikit-learn`

## 📚 Files Modified

1. `train_improved.py` - Enhanced training script
2. `working_kannada_app.py` - Flask app with TTS
3. `src/data/dataset.py` - Enhanced dataset loader
4. `src/utils/transforms.py` - Enhanced preprocessing
5. `templates/optimized_index.html` - Updated UI

## 📖 Additional Documentation

- `TRAINING_GUIDE.md` - Detailed training instructions
- `PROJECT_UPDATES.md` - Summary of all updates



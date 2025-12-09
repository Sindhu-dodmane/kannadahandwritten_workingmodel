# Quick Start Guide - Kannada Character Recognition

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Retrain Model (Recommended)
The improved model with attention and BiLSTM needs to be trained:
```bash
python train_improved.py --data_dir data --epochs 50 --batch_size 64 --use_improved_model
```

**Key Training Features:**
- Automatic class weighting for ottaksharas
- Ottakshara-specific accuracy metrics
- Spatial attention and BiLSTM for better recognition

### 3. Run the Application
```bash
python working_kannada_app.py
```

Then open your browser to: `http://localhost:5000`

## 🎨 New UI Features

### Drawing Canvas
1. Click "✏️ Draw Character" in the sidebar
2. Draw a Kannada character on the canvas
3. Click "Recognize" to get predictions
4. Click "Clear" to start over

### Dark Mode
- Click the "🌙 Dark Mode" button in the header
- Toggle between light and dark themes

### Top-5 Predictions
- After recognition, see the top 5 predictions with confidence scores
- Visual confidence bar shows prediction certainty

### Prediction History
- Right sidebar shows last 20 predictions
- Click any history item to view it again
- Shows timestamp and confidence

### Text-to-Speech
- Click "🔊 Hear Pronunciation" after recognition
- Automatically converts Kannada text to speech
- Works with gTTS (requires internet) or pyttsx3 (offline)

## 🔧 Key Improvements

### For Ottakshara Recognition

1. **Gentler Preprocessing**
   - Preserves fine details like virama marks
   - Reduced blur and thresholding
   - Better resize interpolation

2. **Enhanced Model**
   - Spatial attention mechanism
   - BiLSTM for sequence modeling
   - Better feature extraction

3. **Class Weighting**
   - Automatically boosts ottakshara classes
   - Helps with imbalanced datasets

## 📊 Monitoring Training

Watch for these metrics during training:
```
Epoch  1/50 | Train Loss: 2.3456 Acc: 0.8234 | Val Loss: 1.9876 Acc: 0.8543
  Ottakshara Acc: 0.7234 (1234 samples) | Regular Acc: 0.9123 (4567 samples)
```

- **Ottakshara Acc**: Accuracy specifically for conjunct characters
- **Regular Acc**: Accuracy for regular characters
- Both should improve over time

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `checkpoints/best_improved.pt` exists
- Check that model architecture matches (should have BiLSTM)

### Low Ottakshara Accuracy
- Check if class weighting is working (see training logs)
- Verify preprocessing is preserving fine details
- Consider increasing training epochs

### Drawing Canvas Not Working
- Check browser console for errors
- Ensure JavaScript is enabled
- Try clearing browser cache

### TTS Not Working
- For gTTS: Check internet connection
- For pyttsx3: May have limited Kannada support
- Check browser audio permissions

## 📝 File Structure

```
working_model/
├── src/
│   ├── models/
│   │   └── cnn.py          # Model architecture (with attention + BiLSTM)
│   ├── utils/
│   │   └── transforms.py   # Preprocessing (gentler for ottaksharas)
│   └── data/
│       └── dataset.py      # Dataset loading
├── templates/
│   └── optimized_index.html  # Modern UI with drawing canvas
├── train_improved.py       # Training script (with class weighting)
├── working_kannada_app.py  # Flask application
└── IMPROVEMENTS_SUMMARY.md # Detailed changes documentation
```

## 🎯 Next Steps

1. **Train the Model**
   - Use the improved training script
   - Monitor ottakshara metrics
   - Adjust epochs/batch size as needed

2. **Test Ottaksharas**
   - Use your provided ottakshara samples
   - Check if recognition improves
   - Compare with previous model

3. **Fine-tune if Needed**
   - Load existing checkpoint
   - Continue training with lower learning rate
   - Focus on ottakshara classes

## 💡 Tips

- **Drawing**: Use smooth strokes, don't lift pen too often
- **Images**: Higher resolution images work better
- **History**: Use history to compare predictions
- **Dark Mode**: Easier on eyes for long sessions

## 📞 Support

For issues or questions:
1. Check `IMPROVEMENTS_SUMMARY.md` for detailed changes
2. Review training logs for ottakshara metrics
3. Test with known ottakshara samples

---

**Note**: The model needs to be retrained with the new architecture to see improvements. The old checkpoint may not work with the new BiLSTM architecture.


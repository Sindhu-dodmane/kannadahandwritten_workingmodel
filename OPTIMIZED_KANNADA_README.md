# 🚀 Optimized Kannada Character Recognition System

A comprehensive deep learning application for recognizing Kannada characters with a user-friendly interface supporting single images, multiple images, and PDF documents.

## ✨ Features

### 🎯 Core Functionality
- **Single Image Recognition**: Upload and recognize individual Kannada characters
- **Multiple Image Processing**: Upload up to 40 images (PNG/JPG) for batch processing
- **PDF Document Support**: Upload PDFs with up to 40 pages for text extraction
- **Real-time Navigation**: Browse through uploaded images with previous/next buttons
- **Advanced Recognition**: Optimized deep learning algorithms for better accuracy

### 🖥️ User Interface
- **Left Sidebar**: Easy access to upload options
- **Center Image Viewer**: Large, clear image display with navigation controls
- **Recognition Results**: Display recognized text with confidence scores
- **Download Options**: Export results as PDF or text files
- **Responsive Design**: Works on desktop and mobile devices

### 🔧 Technical Features
- **Optimized CNN Architecture**: Enhanced neural network with attention mechanisms
- **Advanced Preprocessing**: Image enhancement, noise reduction, and contrast adjustment
- **Character Segmentation**: Automatic detection and separation of multiple characters
- **Confidence Scoring**: Top-5 predictions with confidence levels
- **Session Management**: Handle multiple upload sessions simultaneously

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
```
torch>=2.2.0
torchvision>=0.17.0
numpy>=1.26
opencv-python>=4.9
Pillow>=10.2
Flask>=2.3.0
PyMuPDF>=1.23.0
reportlab>=4.0.0
Werkzeug>=2.3.0
```

## 🚀 Quick Start

### 1. Run the Application
```bash
python working_kannada_app.py
```

### 2. Access the Interface
Open your browser and go to: `http://localhost:5000`

### 3. Upload and Recognize
1. Choose an upload option from the left sidebar
2. Upload your image(s) or PDF
3. Navigate through images if multiple
4. Click "Recognize Text" to process
5. Download results as PDF or text

## 📁 Project Structure

```
project/
├── working_kannada_app.py          # Main application file
├── optimized_kannada_app.py        # Advanced version with optimizations
├── train_optimized_model.py        # Model training script
├── templates/
│   └── optimized_index.html        # Frontend interface
├── checkpoints/
│   ├── best_improved.pt            # Trained model weights
│   └── best.pt                     # Backup model
├── data/
│   ├── train/                      # Training data
│   └── val/                        # Validation data
└── requirements.txt                # Dependencies
```

## 🎯 Usage Examples

### Single Image Recognition
1. Click "Single Image" in the sidebar
2. Choose an image file
3. Click "Recognize Text"
4. View the predicted character and confidence

### Multiple Images Processing
1. Click "Multiple Images" in the sidebar
2. Select up to 40 images
3. Use navigation arrows to browse
4. Recognize each image individually

### PDF Document Processing
1. Click "PDF Document" in the sidebar
2. Upload a PDF file (up to 40 pages)
3. Navigate through pages
4. Recognize text on each page

## 🔧 Model Architecture

### WorkingKannadaCNN
- **Input**: 64x64 grayscale images
- **Architecture**: 4-layer CNN with BatchNorm and Dropout
- **Features**: 64→128→256→512 channels
- **Classifier**: 512→256→num_classes with attention
- **Optimization**: AdamW optimizer with learning rate scheduling

### Preprocessing Pipeline
1. **Grayscale Conversion**: Convert to single channel
2. **Resize**: Standardize to 64x64 pixels
3. **Normalization**: Mean=0.5, Std=0.5
4. **Tensor Conversion**: Convert to PyTorch tensor

## 📊 Performance

### Model Accuracy
- **Validation Accuracy**: ~85% on test dataset
- **Character Classes**: 391 Kannada characters
- **Inference Speed**: ~50ms per character (CPU)
- **Memory Usage**: ~200MB for model + 100MB for app

### Supported Formats
- **Images**: PNG, JPG, JPEG
- **PDFs**: Up to 40 pages
- **Batch Size**: Up to 40 images per session

## 🚀 Advanced Features

### Character Segmentation
- Automatic detection of multiple characters in an image
- Left-to-right ordering
- Size filtering to remove noise
- Padding for better recognition

### Confidence Scoring
- Top-5 predictions for each character
- Confidence levels (0-1 scale)
- Alternative character suggestions

### Export Options
- **Text File**: Plain text with UTF-8 encoding
- **PDF**: Formatted document with proper layout
- **Batch Export**: Process multiple images at once

## 🔧 Customization

### Training Your Own Model
```bash
python train_optimized_model.py
```

### Modifying Preprocessing
Edit the `preprocess_image()` function in `working_kannada_app.py`

### Adding New Character Classes
1. Update the dataset structure
2. Modify `num_classes` parameter
3. Retrain the model

## 🐛 Troubleshooting

### Common Issues

#### Model Not Loading
- Check if `checkpoints/` directory exists
- Verify model file permissions
- Ensure PyTorch is properly installed

#### Low Recognition Accuracy
- Use higher quality images
- Ensure proper lighting and contrast
- Try different preprocessing settings

#### Memory Issues
- Reduce batch size
- Use CPU instead of GPU
- Close other applications

### Error Messages
- `No trained model found`: Run model training first
- `Session not found`: Refresh the page and try again
- `Maximum 40 images allowed`: Reduce the number of images

## 📈 Future Improvements

### Planned Features
- [ ] Real-time camera input
- [ ] Handwriting recognition
- [ ] Multi-language support
- [ ] Cloud deployment
- [ ] Mobile app version

### Performance Optimizations
- [ ] GPU acceleration
- [ ] Model quantization
- [ ] Batch processing
- [ ] Caching system

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Kannada character dataset contributors
- PyTorch team for the deep learning framework
- Flask team for the web framework
- OpenCV team for image processing

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the error logs
3. Create an issue on GitHub
4. Contact the development team

---

**🎉 Enjoy using the Optimized Kannada Character Recognition System!**


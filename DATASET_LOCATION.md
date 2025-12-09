# Dataset Location

## 📁 Dataset Path

The new dataset should be located at:

```
C:\Users\HP\Desktop\Data
```

This path is configured as the default in `train_improved.py`.

## 📂 Dataset Structure

Your dataset should be organized in one of these formats:

### Option 1: Train/Val Split Structure (Recommended)
```
C:\Users\HP\Desktop\Data\
├── train\
│   ├── class1\
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   ├── class2\
│   │   └── ...
│   └── ...
└── val\
    ├── class1\
    │   └── ...
    ├── class2\
    │   └── ...
    └── ...
```

### Option 2: Single Folder Structure (Auto-Split)
```
C:\Users\HP\Desktop\Data\
├── class1\
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class2\
│   └── ...
└── ...
```

If you use Option 2, the training script will automatically split your data into train/val sets (default: 85% train, 15% val).

## 🔧 Changing Dataset Location

If your dataset is in a different location, you can specify it when training:

```bash
python train_improved.py --data_dir "YOUR_PATH_HERE"
```

## 📝 Notes

- Each class should be in its own folder
- The folder name will be used as the class name
- Images can be in formats: JPG, PNG, JPEG, etc.
- The training script automatically detects the number of classes from the folder structure



## Kannada Handwritten Character Recognition (CNN + KNN)

This project builds a high-performance recognizer for handwritten Kannada characters by combining a deep CNN feature extractor with a lightweight KNN classifier on learned embeddings. The hybrid approach delivers strong accuracy with fast adaptation to new classes and robust performance under domain shifts.

### Highlights
- CNN trained with metric-learning friendly head to produce discriminative embeddings
- KNN on embeddings for simple, nonparametric decision boundaries and fast class add/remove
- Strong baselines with practical augmentations and test-time tricks

### Repository structure
```
project/
  src/
    cli/
    data/
    models/
    utils/
  data/
  models/
  checkpoints/
  notebooks/
```

### Dataset
- Option A (folders): Use `ImageFolder` structure: `data/train/<class>/*.png` and `data/val/<class>/*.png`.
- Option B (Kaggle CSV): Place `train.csv`, optional `Dig-MNIST.csv` under `data/`. We'll handle stratified split automatically.

#### Kaggle Kannada Characters dataset (`dhruvildave/kannada-characters`)
1) Download and extract the dataset from Kaggle.
2) Identify the root folder that contains subfolders per character class (e.g., `ಕ`, `ಖ`, ...).
3) Run the preparer to build an ImageFolder split:
```
python -m src.cli.prepare_kannada_chars --src <path_to_kaggle_root> --out data --val_ratio 0.1
```
4) Train and evaluate as usual (folder mode):
```
python -m src.cli.train --data_dir data --out_dir checkpoints --epochs 40 --batch_size 256 --image_size 64
python -m src.cli.extract --data_dir data --checkpoint checkpoints/best.pt --out features
python -m src.cli.knn --train features/train.npz --val features/val.npz --k 5 --metric cosine --weights distance
```

### Quickstart
1) Install
```
pip install -r requirements.txt
```
2) Train CNN and export embeddings
```
python -m src.cli.train --data_dir data --out_dir checkpoints
python -m src.cli.extract --data_dir data --checkpoint checkpoints/best.pt --out features/
```
3) Fit and evaluate KNN
```
python -m src.cli.knn --train features/train.npz --val features/val.npz --k 5
```

#### Using Kaggle Kannada-MNIST CSVs
```
python -m src.cli.train --data_dir data --out_dir checkpoints --kaggle_csv
python -m src.cli.extract --data_dir data --checkpoint checkpoints/best.pt --out features --kaggle_csv
python -m src.cli.knn --train features/train.npz --val features/val.npz --k 5 --metric cosine
```

### Patent-oriented notes (non-legal)
- Consider novelty in: curriculum and hard-negative mining for Kannada scripts, stroke-order simulation augmentations, adaptive class-balanced KNN with temperature scaling, and uncertainty-aware rejection for OOD glyphs.
- Keep experiment logs and ablations to support claims.
- Consult a patent attorney for prior art search and claim drafting.

### License
TBD by project owner.


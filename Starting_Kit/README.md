# Grain Variety Classification Challenge

**Starting Kit - Quick Guide**

This repository contains the starting kit for **Group 1 (Grain – Generalization)** of the AI-Master Challenge Course (2025–26) at Université Paris-Saclay.

---

## Challenge Overview

**Goal**: Classify wheat grain varieties from images (8 classes)

**Task**: Multi-class classification using machine learning

**Challenge Focus**: Model generalization across different conditions (dates, imaging conditions)

**Data**: Grain images captured using RGB (3 channels) or Hyperspectral (multiple channels) imaging

---

## Quick Start

1. Navigate to `Starting_Kit/` folder
2. Open `README.ipynb` (main notebook)
3. Run cells in order to explore the baseline

**Installation**:
```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

**Optional** (for deep learning): `tensorflow` or `torch`

---

## Data Description

**File Format**: `.npz` files (compressed NumPy format)
- `x`: Grain image (NumPy array) - shape `(252, 252, 3)` for RGB
- `y`: Variety label (integer, 0-7)

**Loading example**:
```python
data = np.load("grain123.npz")
image = data["x"]  # Image array
variety = data["y"]  # Label (integer)
```

**Dataset Options**:
- **RGB** (recommended): 3-channel images, faster, simpler
- **Spectral**: Multi-channel images, more information but larger

**Tip**: Use `max_samples=1000` for quick testing, remove it for full dataset (~26,000 images)

---

## Baseline Model

**Current Baseline**: Random Forest Classifier (scikit-learn)

- Images resized to `(64, 64)` and flattened
- Fast training on CPU (no GPU required)
- Simple preprocessing: resize, normalize, standardize

**Improvement Ideas**:
- Deep learning models (CNN, ResNet, EfficientNet, ViT)
- Data augmentation (rotation, flip, zoom)
- Better preprocessing and hyperparameter tuning
- Use full spectral data (all bands)

---

## Evaluation Metrics

Three metrics are computed:
1. **Accuracy** (primary for leaderboard)
2. **F1-Score (Macro)** - balanced metric
3. **Cohen's Kappa** - agreement beyond chance

---

## Submission

Submit your model on **Codabench**:
- Create `model.py` with `Model` class (see `Competition_Bundle/sample_code_submission/model.py`)
- Include `requirements.txt` with dependencies
- Zip both files and submit

**Model Interface** (required):
```python
class Model:
    def __init__(self):
        # Initialize model
    
    def fit(self, train_data):
        # Train model (receives dict with 'filepaths' and 'y')
    
    def predict(self, test_data):
        # Return predictions (1D numpy array of encoded classes)
```

---

## Contact

- **Challenge Leader**: Lubin LONGUEPEE - lubin.longuepee@gmail.com
- **GitHub**: https://github.com/md-naim-hassan-saykat/grain-1-generalization-ai-challenge

---

**Good luck! 🚀**

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
3. Run cells in order to:
   - generate a tiny synthetic dataset (no need to download the full data),
   - explore basic visualizations,
   - train and evaluate the baseline model.

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

**Loading example** (real data):
```python
data = np.load("grain123.npz")
image = data["x"]  # Image array
variety = data["y"]  # Label (integer)
```

**Dataset Options**:
- **RGB** (recommended): 3-channel images, faster, simpler
- **Spectral**: Multi-channel images, more information but larger

For convenience, the starting kit can generate a **small synthetic dataset** so that the notebook runs end-to-end without the real data:

```python
from functions import generate_sample_data

# This will create a folder `sample_data/` with a few .npz files
generate_sample_data(output_dir="sample_data", n_samples=16)
```

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

You can also let the starting kit **create the submission zip automatically** once you have your `model.py` and `requirements.txt` in the current folder:

```python
from functions import create_submission_zip

create_submission_zip(
    model_path="model.py",
    requirements_path="requirements.txt",
    output_zip="submission.zip",
)
```

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
- **GitHub**: https://github.com/LubinLgp/CreationAiChallenge_Group1

---

**Good luck! 🚀**

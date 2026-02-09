# Data

## Dataset Overview

- **8 wheat grain varieties** (labeled 2-9)
- **Images collected**: 2020-2021, multiple microplots (32), varying conditions
- **Challenge focus**: Model generalization across different conditions

## Data Format

**File Format**: `.npz` files (compressed NumPy format)

Each file contains:
- `x`: Grain image (NumPy array)
  - RGB: shape `(252, 252, 3)` - 3 channels
  - Spectral: shape `(H, W, N_bands)` - multiple channels
- `y`: Variety label (integer, 2-9)

**Loading example**:
```python
import numpy as np
data = np.load('image.npz')
img = data['x']  # Image array
label = data['y']  # Label (integer)
data.close()
```

## Dataset Splits

- **Training Set**: `input_data/train/` (~21,000+ images)
  - Labels: `reference_data/train_labels.json`
- **Test Set**: `input_data/test/` (~5,000+ images)
  - Labels: **NOT provided** (held out for evaluation)

## Download Instructions

**From Codabench Resources** (Recommended):
1. Go to **Resources** tab
2. Download `training_data.zip`
3. Extract: images in `train/`, labels in `train_labels.json`

**From GitHub**:
- Clone repository and navigate to `Competition_Bundle/input_data/train/`

## Important Notes

- **Training data**: Available for download
- **Test data**: NOT available - only accessible during submission
- **Do NOT use external datasets** - Only provided data allowed
- **Data is for educational purposes only** - Do not share

## Preprocessing Tips

- **Resize**: Images may have different sizes
- **Normalize**: Pixel values to [0, 1] or standardize
- **Data Augmentation**: Rotation, flipping, brightness adjustment
- **Memory**: Use batch loading (don't load all images at once)

---

**For code examples, see the Starting Kit notebook.**

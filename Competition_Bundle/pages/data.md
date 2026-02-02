# Data

## Overview
***

This page describes the grain image dataset used in the **Grain Variety Classification Challenge**. Understanding the data structure, format, and characteristics is essential for developing effective models.

## Dataset Description
***

### Grain Types
The dataset contains images of **8 different wheat grain varieties**, labeled as varieties **2 through 9** (8 classes total).

### Data Collection
- **Source**: Grain images collected from multiple microplots
- **Period**: 2020-2021
- **Imaging Conditions**: Varying camera parameters, lighting, and environmental conditions
- **Purpose**: Educational challenge focusing on model generalization

### Key Challenge: Generalization
The dataset is designed to test **model generalization** across:
- **32 different microplots** with varying growing conditions
- **Multiple acquisition dates** (2020-2021) with different lighting/environmental conditions
- **Different imaging setups** (varying camera parameters, lighting, etc.)

A model that only works on training data but fails on new conditions will not perform well!

## Data Format
***

### File Structure
Each grain image is stored as a **NumPy compressed file (.npz)** with the following structure:

```
image_file.npz
  └── 'x': numpy array containing the image data
```

### Image Characteristics
- **Format**: NumPy arrays stored in .npz files
- **Dimensions**: Variable (typically 252×252 pixels or similar)
- **Channels**: 
  - **RGB images**: 3 channels (Red, Green, Blue)
  - **Hyperspectral images**: Multiple spectral bands (more information than RGB)
- **Data Type**: Typically float32 or uint8
- **Value Range**: Usually [0, 255] for uint8 or [0.0, 1.0] for normalized float32

### Loading Images
To load an image from a .npz file:

```python
import numpy as np

# Load image
data = np.load('path/to/image.npz')
img = data['x']  # Image is stored under key 'x'
data.close()  # Important: close file to free memory

# Image shape: (height, width, channels) or (height, width)
print(f"Image shape: {img.shape}")
```

### Metadata in Filenames
Image filenames may contain metadata such as:
- Grain variety (label)
- Microplot identifier
- Acquisition date
- Imaging conditions

**Note**: Labels are provided separately in label files (see below).

## Dataset Splits
***

The dataset is divided into three splits:

### 1. Training Set
- **Purpose**: Train your model
- **Location**: `input_data/train/`
- **Labels**: Provided in `reference_data/train_labels.json`
- **Usage**: Use this data to learn patterns and train your model
- **Size**: ~21,000+ images (exact number may vary)

### 2. Validation Set (Development Phase)
- **Purpose**: Validate and tune your model during development
- **Location**: Part of training data (you can create your own split)
- **Usage**: Evaluate model performance during development
- **Note**: You can split training data into train/validation for local testing

### 3. Test Set (Final Evaluation)
- **Purpose**: Final evaluation of your model
- **Location**: `input_data/test/`
- **Labels**: **NOT provided** (held out for evaluation)
- **Usage**: The platform evaluates your model on this set
- **Size**: ~5,000+ images (exact number may vary)

## Label Format
***

### Label Files
Labels are provided in JSON format:

- **Training Labels**: `reference_data/train_labels.json`
- **Test Labels**: `reference_data/test_labels.json` (for reference only, not available during submission)

### Label Encoding
- **Format**: Integer labels from 2 to 9 (8 classes)
- **Example**: A grain of variety 3 has label `3`
- **Classes**: 2, 3, 4, 5, 6, 7, 8, 9

### Loading Labels
```python
import json

# Load training labels
with open('reference_data/train_labels.json', 'r') as f:
    labels = json.load(f)

# labels is a dictionary: {filename: label}
# Example: {'grain_001.npz': 3, 'grain_002.npz': 5, ...}
```

### Label Distribution
The dataset may have varying class distributions. Check the Starting Kit for class distribution analysis.

## Data Access
***

### During Competition
- **Training Data**: Available for download (see below)
- **Test Data**: Available during submission (ingested by the platform automatically)
- **Labels**: Training labels provided, test labels held out

### Download Instructions

**Option 1: From Codabench Resources (Recommended)**
1. Go to the **Resources** tab on this competition page
2. Download `training_data.zip` (if available)
3. Extract the zip file
4. Training images are in the `train/` folder
5. Training labels are in `train_labels.json`

**Option 2: From GitHub Repository**
1. Clone the repository: `git clone [repository-url]`
2. Navigate to `Competition_Bundle/input_data/train/`
3. Training labels are in `Competition_Bundle/reference_data/train_labels.json`

**Option 3: Direct Download**
- If the data is hosted elsewhere, check the Data page for direct download links

### Important Notes
- **Training Data**: You can download and use it locally for development
- **Test Data**: NOT available for download - only accessible during submission evaluation
- **Training Labels**: Included with training data download
- **Test Labels**: NOT provided (held out for evaluation)

### Data Size
- **Total Size**: ~XXX GB (exact size may vary)
- **Number of Files**: ~26,000+ image files
- **Storage**: Ensure sufficient disk space for local development

## Data Preprocessing Recommendations
***

### Image Preprocessing
- **Resizing**: Images may have different sizes - consider resizing to a fixed size
- **Normalization**: Normalize pixel values to [0, 1] or standardize (mean=0, std=1)
- **Data Augmentation**: Consider augmentation to improve generalization:
  - Rotation
  - Flipping
  - Brightness/contrast adjustment
  - Color jittering

### Memory Considerations
- **Large Dataset**: The full dataset is large - use data generators/loaders
- **Batch Loading**: Load images in batches rather than all at once
- **File Paths**: The ingestion program provides file paths, not loaded images (to save memory)

## Data Quality Notes
***

- **Image Quality**: Images are captured under varying conditions - models should be robust
- **Class Balance**: Check class distribution - may require balancing techniques
- **Missing Data**: All provided images are valid - no missing data expected
- **Noise**: Some images may have noise or artifacts - robust preprocessing recommended

## Important Reminders
***

1. **Do NOT use external datasets** - Only use provided data
2. **Do NOT share data** - Data is for educational purposes only
3. **Respect data license** - Delete data after challenge ends
4. **Test locally first** - Validate your data loading pipeline before submitting

## Getting Started
***

1. **Download the Starting Kit** (see Starting Kit page)
2. **Explore the data** using the provided notebook
3. **Check data loading code** in the baseline model
4. **Analyze class distribution** and image characteristics
5. **Develop your preprocessing pipeline**

## Questions?
***

If you have questions about the data:
- Check the Starting Kit notebook for examples
- Review the baseline model code for data loading patterns
- Contact organizers through Codabench or GitHub Issues

---

**For code examples on loading and preprocessing data, see the Starting Kit.**

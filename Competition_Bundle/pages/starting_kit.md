# Starting Kit

## Overview
***

Welcome to the **Starting Kit** for the Grain Variety Classification Challenge! This page provides everything you need to get started, including code examples, baseline models, and submission instructions.

## What is a Starting Kit?
***

A Starting Kit is a collection of resources to help you:
- **Understand the problem**: Learn about the data, task, and evaluation
- **Get started quickly**: Use provided code and baseline models
- **Test your setup**: Verify your environment works correctly
- **Submit successfully**: Follow the correct submission format

## Starting Kit Contents
***

### 1. Jupyter Notebook (`README.ipynb`)
A comprehensive notebook that includes:
- **Data Loading**: Code to load and explore the grain image dataset
- **Data Visualization**: Visualizations of class distribution, metadata analysis
- **Baseline Model**: A working baseline model implementation
- **Model Training**: Example training pipeline
- **Evaluation**: Code to evaluate your model locally
- **Submission Preparation**: Instructions for preparing your submission

**Location**: Available in the GitHub repository under `Starting_Kit/README.ipynb`

### 2. Baseline Model (`model.py`)
A working baseline model that you can:
- **Use as-is**: Submit it to test the submission process
- **Modify**: Improve it with better algorithms or preprocessing
- **Learn from**: Understand the required interface and data format

**Features**:
- Uses scikit-learn (Random Forest) - no deep learning required
- Handles data loading from file paths (memory-efficient)
- Includes proper preprocessing (resizing, normalization, scaling)
- Well-documented code with explanations

**Location**: `Competition_Bundle/sample_code_submission/model.py`

### 3. Sample Submission (`submission.zip`)
A ready-to-submit zip file containing:
- `model.py`: The baseline model
- `requirements.txt`: Python dependencies

**Purpose**: 
- Test the submission process
- Verify your environment works
- Understand the expected format

**How to Use**:
1. Download `submission.zip` from the Resources page (or create it using `create_submission_zip.py`)
2. Upload it to Codabench in the "My Submissions" tab
3. Verify it runs successfully

## Getting Started
***

### Step 1: Set Up Your Environment

1. **Install Python 3.9+** (if not already installed)
2. **Install Required Packages**:
   ```bash
   pip install numpy>=1.21.0 scikit-learn>=1.0.0 pandas matplotlib jupyter
   ```
3. **Clone or Download the Repository**:
   ```bash
   git clone [repository-url]
   cd grain-1-generalization-ai-challenge
   ```

### Step 2: Explore the Data

1. **Open the Starting Kit Notebook**:
   ```bash
   cd Starting_Kit
   jupyter notebook README.ipynb
   ```
2. **Run the Data Loading Cells**:
   - Load and explore the dataset
   - Visualize class distribution
   - Understand data structure
3. **Check Data Format**:
   - See how images are stored (.npz files)
   - Understand label encoding
   - Review metadata structure

### Step 3: Run the Baseline Model

1. **Test the Baseline Locally**:
   - Use the notebook to train the baseline model
   - Evaluate on validation data
   - Understand the training process

2. **Test Submission Format**:
   - Create a submission zip (see below)
   - Test locally if possible
   - Prepare for Codabench submission

### Step 4: Improve the Model

1. **Understand the Baseline**:
   - Read the `model.py` code
   - Understand preprocessing steps
   - See what can be improved

2. **Make Improvements**:
   - Try different algorithms (CNN, XGBoost, etc.)
   - Add data augmentation
   - Optimize hyperparameters
   - Implement ensemble methods

3. **Validate Locally**:
   - Split training data into train/validation
   - Evaluate your improved model
   - Compare with baseline

### Step 5: Submit Your Model

1. **Prepare Your Submission**:
   - Ensure `model.py` follows the required interface
   - Create `requirements.txt` with all dependencies
   - Test locally if possible

2. **Create Submission Zip**:
   ```bash
   cd Competition_Bundle
   python3 create_submission_zip.py
   ```
   This creates `submission.zip` with your model.

3. **Submit on Codabench**:
   - Go to "My Submissions" tab
   - Upload `submission.zip`
   - Wait for evaluation
   - Check results on leaderboard

## Submission Format
***

### Required Files

Your submission zip must contain:

1. **`model.py`** (REQUIRED):
   - Must contain a `Model` class
   - Must have `__init__()`, `fit()`, and `predict()` methods
   - See baseline model for example

2. **`requirements.txt`** (OPTIONAL but recommended):
   - List all Python packages your model needs
   - One package per line: `package_name>=version`
   - Example:
     ```
     numpy>=1.21.0
     scikit-learn>=1.0.0
     tensorflow>=2.10.0
     ```

### Model Interface

Your `Model` class must implement:

```python
class Model:
    def __init__(self):
        """Initialize the model (no parameters)"""
        pass
    
    def fit(self, train_data):
        """
        Train the model.
        
        Args:
            train_data: dict with keys:
                - 'filepaths': list of .npz file paths
                - 'y': numpy array of labels (encoded as integers)
        """
        pass
    
    def predict(self, test_data):
        """
        Make predictions.
        
        Args:
            test_data: dict with key:
                - 'filepaths': list of .npz file paths
        
        Returns:
            numpy array of predicted labels (encoded as integers)
        """
        pass
```

### Creating a Submission Zip

**Option 1: Using the provided script** (recommended):
```bash
cd Competition_Bundle
python3 create_submission_zip.py
```

**Option 2: Manual creation**:
```bash
# Create zip with model.py and requirements.txt at root
zip submission.zip model.py requirements.txt
```

**Important**: 
- `model.py` and `requirements.txt` must be at the **root** of the zip
- Do NOT include folders or subdirectories
- Test your zip locally if possible

## Baseline Model Details
***

### Algorithm
- **Type**: Random Forest Classifier (scikit-learn)
- **Trees**: 50 (can be increased for better performance)
- **Max Depth**: 10 (prevents overfitting)
- **No GPU Required**: Fast training on CPU

### Preprocessing
1. **Load Images**: From .npz files (one by one to save memory)
2. **Resize**: Center crop to 64×64 pixels
3. **Flatten**: Convert 2D/3D images to 1D vectors
4. **Normalize**: Pixel values to [0, 1]
5. **Standardize**: Features to mean=0, std=1

### Performance
- **Baseline Accuracy**: ~XX% (check leaderboard for current baseline)
- **Training Time**: ~X minutes (depends on hardware)
- **Memory Usage**: Low (loads images in batches)

### Improving the Baseline
- **Increase trees**: More trees = better performance (but slower)
- **Add preprocessing**: Better resizing, augmentation, feature extraction
- **Try deep learning**: CNN, ResNet, EfficientNet, etc.
- **Ensemble methods**: Combine multiple models
- **Hyperparameter tuning**: Optimize Random Forest parameters

## Resources
***

### Code Repository
- **GitHub**: [grain-1-generalization-ai-challenge](https://github.com/md-naim-hassan-saykat/grain-1-generalization-ai-challenge)
- **Starting Kit**: `Starting_Kit/README.ipynb`
- **Baseline Model**: `Competition_Bundle/sample_code_submission/model.py`

### Documentation
- **Overview Page**: See competition overview and timeline
- **Data Page**: Detailed data description and format
- **Evaluation Page**: Evaluation metrics explanation
- **Terms Page**: Competition rules and terms

### Example Competitions
For inspiration, check these Codabench competitions:
- [Competition 1145](https://www.codabench.org/competitions/1145/)
- [Competition 2044](https://www.codabench.org/competitions/2044/)

## Troubleshooting
***

### Common Issues

1. **"ModuleNotFoundError"**:
   - Add missing package to `requirements.txt`
   - Check package name spelling

2. **"Model not trained"**:
   - Ensure `fit()` is called before `predict()`
   - Check that model is saved in `__init__` or `fit()`

3. **"File not found"**:
   - Check file paths in your code
   - Ensure images are loaded from provided file paths

4. **"Memory error"**:
   - Use data generators (load images in batches)
   - Don't load all images at once
   - See baseline model for example

5. **"Submission failed"**:
   - Check zip structure (model.py at root)
   - Verify model interface matches requirements
   - Check Codabench logs for error messages

### Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Codabench**: Contact organizers through platform
- **Email**: Contact course instructors if needed

## Next Steps
***

1. ✅ **Download Starting Kit**: Get the notebook and baseline model
2. ✅ **Explore Data**: Understand the dataset structure
3. ✅ **Run Baseline**: Test the submission process
4. ✅ **Improve Model**: Develop your own solution
5. ✅ **Submit**: Upload your improved model
6. ✅ **Iterate**: Improve based on leaderboard results

---

**Good luck! We look forward to seeing your creative solutions! 🚀**

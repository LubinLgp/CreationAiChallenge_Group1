# Starting Kit

## Contents

1. **Jupyter Notebook** (`README.ipynb`): Data loading, visualization, baseline training, evaluation
2. **Baseline Model** (`model.py`): Random Forest classifier (scikit-learn)
3. **Sample Submission** (`submission.zip`): Ready-to-submit baseline

**Location**: GitHub repository `Starting_Kit/` folder

## Quick Start

1. **Install packages**:
   ```bash
   pip install numpy scikit-learn pandas matplotlib jupyter
   ```

2. **Open notebook**:
   ```bash
   cd Starting_Kit
   jupyter notebook README.ipynb
   ```

3. **Run cells** to explore data and baseline model

4. **Test submission**: Create zip and submit on Codabench

## Submission Format

Your submission zip must contain:

- **`model.py`** (REQUIRED): Model class with `__init__()`, `fit()`, `predict()` methods
- **`requirements.txt`** (OPTIONAL): Python dependencies

**Important**: Files must be at the **root** of the zip (no folders)

### Model Interface

```python
class Model:
    def __init__(self):
        """Initialize model (no parameters)"""
        pass
    
    def fit(self, train_data):
        """
        Train model.
        train_data: dict with 'filepaths' (list) and 'y' (numpy array)
        """
        pass
    
    def predict(self, test_data):
        """
        Make predictions.
        test_data: dict with 'filepaths' (list)
        Returns: numpy array of predicted labels (integers)
        """
        pass
```

## Creating Submission

**Using script** (recommended):
```bash
cd Competition_Bundle
python3 create_submission_zip.py
```

**Manual**:
```bash
zip submission.zip model.py requirements.txt
```

## Baseline Model

- **Algorithm**: Random Forest (scikit-learn)
- **Preprocessing**: Resize to 64×64, flatten, normalize, standardize
- **No GPU required**: Fast training on CPU

**Improvement ideas**: CNN, data augmentation, hyperparameter tuning, ensemble methods

## Resources

- **GitHub**: https://github.com/LubinLgp/CreationAiChallenge_Group1/tree/main/
- **Baseline**: `Competition_Bundle/sample_code_submission/model.py`
- **Notebook**: `Starting_Kit/README.ipynb`

## Troubleshooting

- **ModuleNotFoundError**: Add package to `requirements.txt`
- **Memory error**: Load images in batches (see baseline model)
- **Submission failed**: Check zip structure (files at root), verify model interface

---

**For detailed examples and code, see the Starting Kit notebook.**

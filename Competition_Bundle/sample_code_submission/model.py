"""
================================================================================
BASELINE MODEL - Grain Classification Challenge
================================================================================

This file contains a baseline model implementation for the Grain Classification
Challenge. This is a simple but functional baseline that participants can use
as a starting point or reference.

WHAT IS THIS FILE?
------------------
This is the model file that participants must submit. It contains a Model class
that implements a machine learning classifier for grain type classification.

WHAT MUST PARTICIPANTS PROVIDE?
-------------------------------
Participants MUST provide a Model class with exactly these 3 methods:
  1. __init__(self): Initializes the model (no parameters required)
  2. fit(self, train_data): Trains the model on training data
  3. predict(self, test_data): Makes predictions on test data

The Model class will be instantiated, trained, and used for prediction by the
competition's ingestion program. The exact interface is described below.

WHAT IS OPTIONAL?
-----------------
- The specific algorithm (Random Forest, Neural Network, etc.) - participants
  can use any method they want
- The preprocessing steps (resizing, normalization, feature extraction, etc.)
- Additional helper methods or classes
- The libraries used (as long as they're listed in requirements.txt)

WHAT CAN PARTICIPANTS CHANGE?
-----------------------------
Everything! This is just a baseline. Participants can:
- Replace Random Forest with any other algorithm (CNN, SVM, etc.)
- Change preprocessing (different resize, data augmentation, etc.)
- Add feature engineering
- Use different libraries (TensorFlow, PyTorch, XGBoost, etc.)
- Optimize hyperparameters
- Implement ensemble methods
- Add any other improvements

IMPORTANT NOTES:
----------------
- The fit() method receives a dictionary with 'filepaths' (list of .npz file
  paths) and 'y' (encoded labels). Images are NOT pre-loaded to save memory.
- The predict() method receives a dictionary with 'filepaths' (list of .npz
  file paths). It must return a 1D numpy array of encoded class predictions.
- Labels are already encoded as integers (0, 1, 2, ...). No need to decode.
- Images are stored in .npz files. Load with: data = np.load(filepath); img = data['x']
- The model should handle images of different sizes (this baseline resizes to 64x64)

BASELINE IMPLEMENTATION:
------------------------
This baseline uses:
- scikit-learn's RandomForestClassifier (no deep learning, fast to train)
- Simple preprocessing: resize to 64x64, flatten, normalize, standardize
- 50 trees, max_depth=10 (small for speed, can be increased for better performance)

This baseline achieves reasonable performance but can be significantly improved
with better architectures, preprocessing, or hyperparameter tuning.

================================================================================
"""

# ----------------------------------------
# Imports
# ----------------------------------------
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# ----------------------------------------
# Model Class
# ----------------------------------------
class Model:
    """
    Baseline Model class for grain classification.
    
    This class implements a simple Random Forest classifier as a baseline.
    Participants can modify or completely replace this implementation.
    
    The class must have these three methods:
    - __init__(): Initialize the model (no parameters)
    - fit(train_data): Train the model on training data
    - predict(test_data): Make predictions on test data
    """
    
    def __init__(self):
        """
        Initialize the baseline model.
        
        This constructor sets up the model architecture and preprocessing.
        No parameters are required - the ingestion program will call this
        without arguments.
        
        Attributes initialized:
        - model: The Random Forest classifier (None until fit() is called)
        - scaler: StandardScaler for feature normalization
        - num_classes: Number of classes (determined during training)
        - img_size: Target image size for resizing (64, 64) to speed up training
        """
        print("[*] - Initializing Baseline Classifier (scikit-learn, no TensorFlow)")
        self.model = None
        self.scaler = StandardScaler()
        self.num_classes = None
        self.img_size = (64, 64)  # Resize to smaller size for speed

    def fit(self, train_data):
        """
        Train the model on the provided training data.
        
        This method loads training images from file paths, preprocesses them,
        and trains a Random Forest classifier. The training process includes:
        1. Loading images from .npz files
        2. Resizing images to a fixed size (64x64)
        3. Flattening images to 1D vectors
        4. Normalizing pixel values to [0, 1]
        5. Standardizing features (mean=0, std=1)
        6. Training Random Forest classifier
        
        Parameters
        ----------
        train_data : dict
            Dictionary containing training data. Must have one of:
            
            - 'filepaths': list of str
                List of file paths to .npz files containing training images.
                Each .npz file contains an image under the key 'x'.
                This is the format used by the competition ingestion program.
                
            - 'X': numpy.ndarray (optional, for backward compatibility)
                Pre-loaded training images as a numpy array.
                Shape: (n_samples, height, width, channels) or (n_samples, height, width)
                
            - 'y': numpy.ndarray
                Training labels encoded as integers (0, 1, 2, ...).
                Shape: (n_samples,)
        
        Raises
        ------
        ValueError
            If train_data doesn't contain 'X' or 'filepaths', or if 'y' is missing.
        
        Notes
        -----
        - Images are loaded one by one to save memory (important for large datasets)
        - Images are resized by taking a center crop if they're larger than img_size
        - The number of classes is automatically determined from unique labels in 'y'
        - Training progress is printed to stdout (visible in Codabench logs)
        """
        print("[*] - Training Baseline Classifier on the train set")
        
        # =====================================================================
        # STEP 1: Load training data
        # =====================================================================
        # The ingestion program provides data as file paths to save memory.
        # We load images one by one from .npz files.
        if 'X' in train_data:
            # Backward compatibility: data already loaded in memory
            X_train = train_data['X']
            y_train = train_data['y']
        elif 'filepaths' in train_data:
            # Competition format: load images from file paths
            filepaths = train_data['filepaths']
            y_train = train_data['y']
            
            print(f"[*] Loading {len(filepaths)} training images...")
            X_train = []
            for i, filepath in enumerate(filepaths):
                # Progress indicator (prints every 1000 images)
                if i % 1000 == 0:
                    print(f"  Loaded {i}/{len(filepaths)} images...")
                
                # Load image from .npz file
                # Each .npz file contains the image under key 'x'
                data = np.load(filepath)
                img = data['x']  # Image stored as 'x' in the .npz file
                data.close()  # Close file to free memory
                
                # =============================================================
                # STEP 2: Preprocess image (resize if needed)
                # =============================================================
                # Images may have different sizes. We resize to a fixed size
                # by taking a center crop. This is a simple approach - participants
                # can improve this with better resizing, padding, or augmentation.
                if img.shape[:2] != self.img_size:
                    h, w = img.shape[:2]
                    target_h, target_w = self.img_size
                    # Center crop: take middle portion of the image
                    start_h = (h - target_h) // 2
                    start_w = (w - target_w) // 2
                    img = img[start_h:start_h+target_h, start_w:start_w+target_w]
                
                # =============================================================
                # STEP 3: Flatten image to 1D vector
                # =============================================================
                # Random Forest needs 1D feature vectors, so we flatten the image.
                # For example, a 64x64x3 RGB image becomes a 12288-dimensional vector.
                img_flat = img.flatten()
                X_train.append(img_flat)
            
            # Convert list to numpy array
            X_train = np.array(X_train, dtype=np.float32)
            print(f"[*] Loaded {len(X_train)} images, shape: {X_train.shape}")
        else:
            raise ValueError("train_data must contain 'X' or 'filepaths'")
        
        # =====================================================================
        # STEP 4: Normalize pixel values to [0, 1]
        # =====================================================================
        # Images may be in [0, 255] range. We normalize to [0, 1] for better
        # numerical stability and convergence.
        if X_train.max() > 1.0:
            X_train = X_train / 255.0
        
        # =====================================================================
        # STEP 5: Determine number of classes
        # =====================================================================
        # The number of classes is automatically determined from unique labels.
        # This is useful for validation and logging.
        self.num_classes = len(np.unique(y_train))
        print(f"[*] Number of classes: {self.num_classes}")
        
        # =====================================================================
        # STEP 6: Ensure images are flattened (safety check)
        # =====================================================================
        # If data was provided as 'X' (already loaded), it might be 3D/4D.
        # We reshape it to 2D: (n_samples, n_features)
        if len(X_train.shape) > 2:
            n_samples = X_train.shape[0]
            X_train = X_train.reshape(n_samples, -1)
        
        # =====================================================================
        # STEP 7: Standardize features (mean=0, std=1)
        # =====================================================================
        # StandardScaler normalizes features to have zero mean and unit variance.
        # This helps Random Forest and many other algorithms perform better.
        print("[*] Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # =====================================================================
        # STEP 8: Train Random Forest classifier
        # =====================================================================
        # Random Forest is a simple but effective baseline. It's fast to train
        # and doesn't require GPU. Participants can replace this with:
        # - Deep learning (CNN, ResNet, EfficientNet, etc.)
        # - Other tree-based methods (XGBoost, LightGBM, etc.)
        # - SVM, k-NN, or any other classifier
        print("[*] Training Random Forest classifier...")
        self.model = RandomForestClassifier(
            n_estimators=50,      # Number of trees (small for speed, increase for better performance)
            max_depth=10,         # Maximum depth of trees (prevents overfitting)
            random_state=42,      # Random seed for reproducibility
            n_jobs=-1,           # Use all CPU cores for parallel training
            verbose=1            # Print progress during training
        )
        self.model.fit(X_train_scaled, y_train)
        print("[*] Training completed")

    def predict(self, test_data):
        """
        Make predictions on test data.
        
        This method loads test images from file paths, applies the same
        preprocessing as during training, and returns class predictions.
        The preprocessing must match exactly what was done in fit() to ensure
        consistent results.
        
        Parameters
        ----------
        test_data : dict
            Dictionary containing test data. Must have one of:
            
            - 'filepaths': list of str
                List of file paths to .npz files containing test images.
                Each .npz file contains an image under the key 'x'.
                This is the format used by the competition ingestion program.
                
            - 'X': numpy.ndarray (optional, for backward compatibility)
                Pre-loaded test images as a numpy array.
                Shape: (n_samples, height, width, channels) or (n_samples, height, width)
        
        Returns
        -------
        y_pred : numpy.ndarray
            1D array of predicted class labels (encoded as integers: 0, 1, 2, ...).
            Shape: (n_samples,)
            The labels are already encoded - no need to decode them.
            The ingestion program will handle label decoding automatically.
        
        Raises
        ------
        ValueError
            If the model hasn't been trained yet (fit() not called), or if
            test_data doesn't contain 'X' or 'filepaths'.
        
        Notes
        -----
        - The preprocessing steps (resize, normalize, scale) must match exactly
          what was done in fit() to ensure the model receives data in the same format.
        - Predictions are returned as integers (int32) as required by the competition.
        - The order of predictions must match the order of filepaths in test_data.
        """
        print("[*] - Predicting test set using Baseline Classifier")
        
        # Check that model has been trained
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # =====================================================================
        # STEP 1: Load test data
        # =====================================================================
        # Same process as in fit(): load images from file paths one by one.
        if 'X' in test_data:
            # Backward compatibility: data already loaded
            X_test = test_data['X']
        elif 'filepaths' in test_data:
            # Competition format: load images from file paths
            filepaths = test_data['filepaths']
            
            print(f"[*] Loading {len(filepaths)} test images...")
            X_test = []
            for i, filepath in enumerate(filepaths):
                # Progress indicator
                if i % 1000 == 0:
                    print(f"  Loaded {i}/{len(filepaths)} images...")
                
                # Load image from .npz file
                data = np.load(filepath)
                img = data['x']
                data.close()
                
                # =============================================================
                # STEP 2: Preprocess image (same as training)
                # =============================================================
                # IMPORTANT: Use the same preprocessing as in fit()!
                if img.shape[:2] != self.img_size:
                    h, w = img.shape[:2]
                    target_h, target_w = self.img_size
                    # Center crop (same as training)
                    start_h = (h - target_h) // 2
                    start_w = (w - target_w) // 2
                    img = img[start_h:start_h+target_h, start_w:start_w+target_w]
                
                # =============================================================
                # STEP 3: Flatten image (same as training)
                # =============================================================
                img_flat = img.flatten()
                X_test.append(img_flat)
            
            X_test = np.array(X_test, dtype=np.float32)
        else:
            raise ValueError("test_data must contain 'X' or 'filepaths'")
        
        # =====================================================================
        # STEP 4: Normalize pixel values (same as training)
        # =====================================================================
        if X_test.max() > 1.0:
            X_test = X_test / 255.0
        
        # =====================================================================
        # STEP 5: Ensure images are flattened (safety check)
        # =====================================================================
        if len(X_test.shape) > 2:
            n_samples = X_test.shape[0]
            X_test = X_test.reshape(n_samples, -1)
        
        # =====================================================================
        # STEP 6: Standardize features (same scaler as training)
        # =====================================================================
        # IMPORTANT: Use transform(), not fit_transform()!
        # The scaler was already fitted on training data in fit().
        # We only transform the test data using the same mean/std.
        X_test_scaled = self.scaler.transform(X_test)
        
        # =====================================================================
        # STEP 7: Make predictions
        # =====================================================================
        # The model predicts class labels (encoded as integers).
        print("[*] Making predictions...")
        y_pred = self.model.predict(X_test_scaled)
        
        # Log prediction statistics
        print(f"[*] Generated {len(y_pred)} predictions")
        print(f"[*] Unique predicted classes: {np.unique(y_pred)}")
        
        # Return predictions as int32 (required format)
        return y_pred.astype(np.int32)

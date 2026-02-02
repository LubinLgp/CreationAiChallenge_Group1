# Evaluation

## Overview
***

This page describes how submissions are evaluated in the **Grain Variety Classification Challenge**. Understanding the evaluation metrics is crucial for developing effective models.

## Evaluation Metrics
***

The competition uses **three primary metrics** to evaluate model performance:

### 1. Accuracy
**Primary Metric** - The main metric used for ranking on the leaderboard.

- **Definition**: Percentage of correctly classified grains
- **Formula**: `Accuracy = (Number of correct predictions) / (Total number of predictions)`
- **Range**: 0.0 to 1.0 (or 0% to 100%)
- **Interpretation**: Higher is better. An accuracy of 0.85 means 85% of grains are correctly classified.

**Why Accuracy?**
- All grain varieties are equally important
- Easy to interpret and compare across submissions
- Directly answers "how many grains are correctly identified"
- Suitable for balanced multi-class classification

### 2. F1-Score (Macro-Averaged)
**Secondary Metric** - Provides insight into per-class performance.

- **Definition**: Macro-averaged F1-score across all classes
- **Formula**: `F1-Macro = (1/N) × Σ F1(class_i)` where N is the number of classes
- **Range**: 0.0 to 1.0
- **Interpretation**: Higher is better. Accounts for both precision and recall per class, then averages.

**Why F1-Score (Macro)?**
- Balances precision and recall for each class
- Macro-averaging treats all classes equally (important for balanced evaluation)
- Reveals if the model performs well across all varieties or only some
- Helps identify if certain grain varieties are harder to classify

### 3. Cohen's Kappa
**Secondary Metric** - Measures agreement beyond chance.

- **Definition**: Measures inter-rater agreement, accounting for agreement by chance
- **Formula**: `κ = (p₀ - pₑ) / (1 - pₑ)` where p₀ is observed agreement and pₑ is expected agreement
- **Range**: -1.0 to 1.0 (typically 0.0 to 1.0 for classification)
- **Interpretation**: 
  - κ < 0: Poor agreement
  - 0.0 ≤ κ < 0.2: Slight agreement
  - 0.2 ≤ κ < 0.4: Fair agreement
  - 0.4 ≤ κ < 0.6: Moderate agreement
  - 0.6 ≤ κ < 0.8: Substantial agreement
  - 0.8 ≤ κ ≤ 1.0: Almost perfect agreement

**Why Cohen's Kappa?**
- Accounts for agreement that could occur by chance
- More informative than accuracy when classes are imbalanced
- Standard metric in classification challenges
- Provides confidence in model performance beyond random guessing

## Additional Metrics (For Analysis)
***

While not used for ranking, the following metrics are computed for detailed analysis:

- **Per-Class Accuracy**: Accuracy for each grain variety individually
- **Confusion Matrix**: Shows which classes are confused with which
- **Precision per Class**: Percentage of predicted positives that are actually positive
- **Recall per Class**: Percentage of actual positives that are correctly identified

These metrics help participants understand model strengths and weaknesses.

## Evaluation Process
***

1. **Submission**: Participants submit their model code (see Starting Kit for format)
2. **Ingestion**: The platform loads your model and runs it on test data
3. **Prediction**: Your model generates predictions for all test images
4. **Scoring**: Predictions are compared against ground truth labels
5. **Metrics Calculation**: All three metrics (Accuracy, F1-Score, Cohen's Kappa) are computed
6. **Leaderboard Update**: Results are displayed on the leaderboard (sorted by Accuracy)

## Important Notes
***

- **Test Set**: The test set is held out and labels are not provided to participants
- **Automatic Evaluation**: Evaluation is performed automatically by the Codabench platform
- **Reproducibility**: The same test set is used for all submissions to ensure fair comparison
- **Final Ranking**: The Final Evaluation Phase leaderboard determines winners (based on Accuracy)
- **Generalization Focus**: Models are evaluated on grains from different conditions (microplots, dates, imaging setups) - this tests true generalization ability

## Understanding Your Results
***

When you receive your evaluation results:

1. **Check Accuracy**: This is your primary score for ranking
2. **Review F1-Score**: If F1 is much lower than Accuracy, your model may struggle with certain classes
3. **Examine Cohen's Kappa**: If Kappa is low relative to Accuracy, your model may be benefiting from class imbalance
4. **Analyze Per-Class Metrics**: Identify which grain varieties are hardest to classify
5. **Compare with Baseline**: The baseline model achieves ~XX% accuracy (see Starting Kit)

## Tips for Improving Scores
***

- **Focus on Generalization**: The challenge emphasizes generalization - ensure your model works across different conditions
- **Balance All Classes**: Don't optimize only for overall accuracy - ensure all varieties are well-classified
- **Use Data Augmentation**: Helps models generalize to different imaging conditions
- **Consider Ensemble Methods**: Combining multiple models often improves performance
- **Validate Locally**: Test your model on validation data before submitting

---

**For more details on the evaluation implementation, see the scoring program in the competition bundle.**

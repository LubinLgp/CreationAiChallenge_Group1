# Evaluation

## Metrics

Three metrics are computed to evaluate submissions:

### 1. Accuracy (Primary Metric)
- **Definition**: Percentage of correctly classified grains
- **Formula**: `Accuracy = (Correct predictions) / (Total predictions)`
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Used for**: Leaderboard ranking

### 2. F1-Score (Macro)
- **Definition**: Macro-averaged F1-score across all classes
- **Why**: Balances precision and recall, treats all classes equally
- **Helps identify**: If model performs well across all varieties or only some

### 3. Cohen's Kappa
- **Definition**: Measures agreement beyond chance
- **Range**: -1.0 to 1.0 (typically 0.0 to 1.0)
- **Why**: Accounts for agreement that could occur by chance, more informative than accuracy alone

## Evaluation Process

1. Submit your model code
2. Platform loads and runs it on test data
3. Predictions are compared against ground truth
4. All three metrics are computed
5. Results displayed on leaderboard (sorted by Accuracy)

## Important Notes

- **Test set labels are NOT provided** (held out for evaluation)
- **Automatic evaluation** on Codabench platform
- **Final ranking** based on Accuracy (Final Evaluation Phase)
- **Generalization focus**: Models tested on grains from different conditions

## Understanding Results

- **Accuracy**: Your primary ranking score
- **F1-Score**: If much lower than Accuracy, model may struggle with certain classes
- **Cohen's Kappa**: If low relative to Accuracy, model may benefit from class imbalance

---

**For detailed metric explanations and tips, see the Starting Kit notebook.**

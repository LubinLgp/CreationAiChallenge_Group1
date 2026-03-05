# Grain Variety Classification Challenge – Group 1

This repository contains the **challenge bundle** and **starting kit** for the *Grain Variety Classification Challenge* (AI Master, Université Paris-Saclay 2025–26).  
The goal is to classify wheat grain varieties (8 classes) from images, with a strong focus on **generalization across microplots, dates and imaging conditions**.

---

## Repository structure (for participants)

Only the parts below are intended for participants and for the public GitHub:

- `Competition_Bundle/`  
  Official Codabench bundle:
  - `competition.yaml`: competition configuration.
  - `ingestion_program/`: code that loads your `Model`, trains it and produces predictions.
  - `scoring_program/`: code that computes metrics (Accuracy, F1-macro, Cohen's κ).
  - `pages/`: markdown pages used on Codabench (overview, data, evaluation, terms, starting kit).
  - `reference_data/`: small metadata and label files used for local checks.
  - `input_data/`: README explaining the expected structure of the data folder.

- `Starting_Kit/`  
  Minimal **starting kit** for new participants:
  - `README.ipynb`: main notebook to explore the problem and run a baseline.
  - `README.md`: quick guide (installation, data format, metrics, submission).
  - `functions.py`: helper functions and classes:
    - data loading and simple visualizations,
    - baseline Random Forest on RGB images,
    - `generate_sample_data(...)` to create a tiny synthetic dataset so the notebook runs without the full data,
    - `create_submission_zip(...)` to build a `submission.zip` (with `model.py` and `requirements.txt`) ready for Codabench.

- `Competition_Bundle/sample_code_submission/`  
  - `model.py`: reference implementation of the **baseline Random Forest** trained on 64×64 RGB crops.
  - `requirements.txt`: minimal dependencies to run the baseline on Codabench.

Other folders (e.g. `methods/`, `report/`, `latex/`, `Grain/`, `Pollinator/`, development artifacts like `.idea/`, `pico.save`, etc.) are internal to the course project and **are not needed** to use the challenge or the starting kit.

---

## Installation

We recommend creating a fresh environment (e.g. with `conda` or `venv`) and installing the basic dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

For deep learning experiments (not required by the baseline), you can additionally install `torch` or `tensorflow`.

---

## Using the starting kit

1. Go to the starting kit folder:
   ```bash
   cd Starting_Kit
   ```
2. Launch Jupyter and open `README.ipynb`:
   ```bash
   jupyter notebook
   ```
3. In the notebook, you can:
   - call `generate_sample_data(output_dir="sample_data", n_samples=16)` to create a tiny synthetic dataset,
   - visualize the data and metadata (class distribution, microplots, etc.),
   - train and evaluate the baseline Random Forest,
   - generate a Codabench submission zip with:
     ```python
     from functions import create_submission_zip
     create_submission_zip("model.py", "requirements.txt", "submission.zip")
     ```

---

## Submitting to Codabench

On Codabench, each submission is a zip file containing at least:

- `model.py` – defines a `Model` class with:
  - `__init__(self)`,
  - `fit(self, train_data)`,
  - `predict(self, test_data)` returning a 1D `numpy` array of predicted class indices.
- `requirements.txt` – Python dependencies needed to import and run your model.

You can either:

- create the zip manually, or
- use the helper in the starting kit:
  ```python
  from functions import create_submission_zip
  create_submission_zip("model.py", "requirements.txt", "submission.zip")
  ```

Then upload `submission.zip` on the Codabench challenge page.

---

## Links

- **INRAE** (data provider): <https://www.inrae.fr>  
- **Challenge on Codabench**: link provided in the course materials and in the paper (Section II-D).\n


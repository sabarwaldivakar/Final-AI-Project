# Predicting Bank Term Deposits — B104 Final Project

Author: Divakar Sabarwal  
Updated: 13 September 2026

## What this is
A lecture-aligned end-to-end machine-learning notebook that predicts whether a contacted customer will subscribe to a bank term deposit. The project uses the UCI "bank-additional-full" dataset (date-ordered May 2008–Nov 2010) and demonstrates data auditing, chronological train/validation/test splitting, preprocessing, a small manual hyperparameter grid, model selection by F1, and final evaluation.

Selected model (final): Logistic Regression, C=0.1, class_weight="balanced".

### Key results (test)
- Precision: 36.82%
- Recall: 77.79%
- F1: 0.4998
- ROC-AUC: 0.6837

## Files
Top-level files and folders:
- Bank_management2.ipynb — Executed Jupyter notebook implementing the full ML workflow (EDA → preprocessing → model selection → final evaluation).
- Bank_management2.html — HTML export of the executed notebook for reading/submission.
- B104_Code_and_Lecture_Guide.md — A study guide that explains design choices and links to lecture material.
- B104_Project_Roadmap.md — Roadmap and completion checklist.
- data/dataset.csv — Local dataset (UCI bank-additional-full.csv); the notebook loads this file with `sep=";"`.
- .gitignore

## Stack / Dependencies
- Language: Python 3 (notebook executed with Python 3.9.x)
- Primary libs used: pandas, NumPy, scikit-learn, matplotlib
- Notebook reports: Python 3.9.6 | pandas 2.3.3 | NumPy 2.0.2 | scikit-learn 1.6.1 (see first code cell output)

Create a requirements.txt capturing the versions above if you want reproducibility, for example:
pip install pandas==2.3.3 numpy==2.0.2 scikit-learn==1.6.1 matplotlib jupyter

Note: the repository references a requirements.txt in the study guide but that file is not present here — consider adding it.

## How to run
1. Clone the repo and ensure the dataset is at `data/dataset.csv`.
2. Open and run the notebook from a Jupyter server or compatible editor. Restart the kernel and run all cells in order (later cells depend on earlier outputs).

Commands:
- Install dependencies (example):
  pip install pandas==2.3.3 numpy==2.0.2 scikit-learn==1.6.1 matplotlib jupyter
- Open the notebook:
  jupyter notebook Bank_management2.ipynb
- Export an executed notebook to HTML:
  python3 -m nbconvert --to html Bank_management2.ipynb

Important:
- The notebook expects `data/dataset.csv` (semicolon-separated).
- Bank_management2.html is a read-only export — edits must be made in the .ipynb and re-exported.
- The notebook was executed in-place and prints the software versions in its first code cell.

## Reproducibility notes & recommendations
- Add a requirements.txt that pins the package versions printed by the notebook.
- If you want deterministic results across environments, consider pinning more packages and adding a short environment setup (venv/conda + requirements).
- Add a LICENSE (none included), and a CONTRIBUTING.md if you accept contributions.
- If you plan to run experiments programmatically, consider adding a small runner script (e.g., run_experiment.py) that loads the notebook configuration or converts key notebook cells into scripts.

## Data & citation
- Dataset: UCI "bank-additional-full.csv" (original source documented in the notebook). The notebook loads the dataset from `data/dataset.csv` using `pd.read_csv(..., sep=";")`.
- The notebook contains discussion about dataset duplicates, splitting by chronological order, and feature selection to avoid prediction-time leakage.

## Limitations / Caveats
- The notebook intentionally excludes `duration` and some current-campaign/economic indicators because they are not guaranteed to be available at the prediction time.
- No requirements.txt or LICENSE present in the repository (recommended to add).
- The dataset rows do not include a customer ID; duplicate removal is by exact row duplicates only and cannot verify unique customers.

## How to cite / Acknowledgements
This work follows lecture material for an ML course (B104). See B104_Code_and_Lecture_Guide.md for references and a lecture map.

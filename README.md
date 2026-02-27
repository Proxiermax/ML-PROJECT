# ML-PROJECT

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Machine Learning project with **11 models** implemented both **from scratch** and using **scikit-learn (lib)**, with side-by-side comparison.

## Models

| #   | Model                                | Type           | Scratch | Lib (sklearn) |
| --- | ------------------------------------ | -------------- | ------- | ------------- |
| 1   | Linear Regression                    | Regression     | ✅      | ✅            |
| 2   | Polynomial Regression                | Regression     | ✅      | ✅            |
| 3   | Multiple Regression                  | Regression     | ✅      | ✅            |
| 4   | Logistic Regression                  | Classification | ✅      | ✅            |
| 5   | Decision Tree                        | Classification | ✅      | ✅            |
| 6   | Random Forest                        | Classification | ✅      | ✅            |
| 7   | SVM                                  | Classification | ✅      | ✅            |
| 8   | Perceptron / SLP                     | Classification | ✅      | ✅            |
| 9   | MLP (Multi-Layer Perceptron)         | Classification | ✅      | ✅            |
| 10  | KNN (Custom Classification)          | Classification | ✅      | ✅            |
| 11  | Clustering (K-Means + Agglomerative) | Clustering     | ✅      | ✅            |

## Project Organization

```
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── data/
│   └── raw/                    <- Raw dataset (computer_prices_all.csv)
├── docs/
├── models/                     <- Saved .pkl model files
├── notebooks/
├── references/
├── reports/
│   └── figures/
└── src/
    ├── __init__.py
    ├── test_predict.py         <- Compare scratch vs lib predictions
    ├── data/
    │   ├── raw_data.py         <- Load raw CSV
    │   ├── regression_data.py  <- Prepare regression features/target
    │   └── classification_data.py <- Prepare classification features/target
    └── modeling/
        ├── evaluation.py       <- Metrics & comparison tables
        ├── scratch/            <- Models built from scratch (numpy only)
        │   ├── linear_regression/
        │   ├── poly_regression/
        │   ├── multiple_regression/
        │   ├── logistic_regression/
        │   ├── decision_tree/
        │   ├── random_forest/
        │   ├── svm/
        │   ├── perceptron/
        │   ├── mlp/
        │   ├── custom_classification/
        │   └── clustering/
        └── lib/                <- Models using scikit-learn
            ├── linear_regression/
            ├── poly_regression/
            ├── multiple_regression/
            ├── logistic_regression/
            ├── decision_tree/
            ├── random_forest/
            ├── svm/
            ├── perceptron/
            ├── mlp/
            ├── custom_classification/
            └── clustering/
```

Each model folder contains:

- `model.py` — Model class (scratch) or factory function (lib)
- `train.py` — Train, evaluate, save, and compare scratch vs lib
- `predict.py` — Load saved model and predict

---

## Setup

```bash
pip install uv
uv sync
```

---

## How to Train

Every scratch `train.py` trains **both** the scratch and lib versions, then prints a **comparison table**.

### Train a single model (scratch + lib comparison)

```bash
# Regression models
uv run python -m src.modeling.scratch.linear_regression.train
uv run python -m src.modeling.scratch.poly_regression.train
uv run python -m src.modeling.scratch.multiple_regression.train

# Classification models
uv run python -m src.modeling.scratch.logistic_regression.train
uv run python -m src.modeling.scratch.decision_tree.train
uv run python -m src.modeling.scratch.random_forest.train
uv run python -m src.modeling.scratch.svm.train
uv run python -m src.modeling.scratch.perceptron.train
uv run python -m src.modeling.scratch.mlp.train
uv run python -m src.modeling.scratch.custom_classification.train

# Clustering
uv run python -m src.modeling.scratch.clustering.train
```

### Train lib version only

```bash
uv run python -m src.modeling.lib.linear_regression.train
uv run python -m src.modeling.lib.logistic_regression.train
# ... same pattern for all models
```

### Example output (comparison table)

```
============================================================
  Comparison: Linear Regression
============================================================
  Metric              Scratch  Lib (sklearn)       Diff
  -----------------------------------------------------
  mse               1234.5678       1200.1234    -34.4444
  rmse                35.1364         34.6429     -0.4935
  mae                 28.1234         27.8901     -0.2333
  r2                   0.8500          0.8600     +0.0100
============================================================
```

---

## How to Test (Predict)

Compare scratch vs lib predictions on custom input:

```bash
uv run python -m src.test_predict
```

You will be prompted to enter feature values separated by commas. The output shows predictions from both scratch and lib models side-by-side:

```
============================================================
  Prediction Comparison: Scratch vs Lib (sklearn)
============================================================

  Linear Regression
    Scratch : 2475.0
    Lib     : 2500.0

  Polynomial Regression
    Scratch : 2450.0
    Lib     : 2475.0

  Multiple Regression
    Scratch : 2500.0
    Lib     : 2525.0

============================================================
```

---

## Saved Models

Models are saved as `.pkl` files in the `models/` directory:

| Version | Naming Pattern          | Example                           |
| ------- | ----------------------- | --------------------------------- |
| Scratch | `{model}_model.pkl`     | `linear_regression_model.pkl`     |
| Lib     | `lib_{model}_model.pkl` | `lib_linear_regression_model.pkl` |

## Run Api

```bash
uv run uvicorn src.main:app --reload
```

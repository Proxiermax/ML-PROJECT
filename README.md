# ML-PROJECT

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Machine Learning project with **13 models** implemented both **from scratch** and using **scikit-learn (lib)**, with side-by-side comparison.

## Models

| #   | Model                                | Type           | Scratch | Lib (sklearn) |
| --- | ------------------------------------ | -------------- | ------- | ------------- |
| 1   | Linear Regression                    | Regression     | ✅      | ✅            |
| 2   | Polynomial Regression                | Regression     | ✅      | ✅            |
| 3   | Multiple Regression                  | Regression     | ✅      | ✅            |
| 4   | MLP Regression                       | Regression     | ✅      | —             |
| 5   | Logistic Regression                  | Classification | ✅      | ✅            |
| 6   | Decision Tree                        | Classification | ✅      | ✅            |
| 7   | Random Forest                        | Classification | ✅      | ✅            |
| 8   | SVM                                  | Classification | ✅      | ✅            |
| 9   | Perceptron / SLP                     | Classification | ✅      | ✅            |
| 10  | MLP (Multi-Layer Perceptron)         | Classification | ✅      | ✅            |
| 11  | KNN (Custom Classification)          | Classification | ✅      | ✅            |
| 12  | Clustering (K-Means + Agglomerative) | Clustering     | ✅      | ✅            |
| 13  | XGBoost                              | Classification | ✅      | —             |

## Project Organization

```
├── LICENSE
├── Makefile
├── README.md
├── pyproject.toml
├── data/
│   └── raw/                        <- Raw dataset (computer_prices_all.csv)
├── docs/
├── models/                         <- Saved .pkl model files
├── notebooks/
├── references/
├── reports/
│   └── figures/
└── src/
    ├── __init__.py
    ├── main.py                     <- FastAPI app for price prediction
    ├── test_predict.py             <- Compare scratch vs lib predictions
    ├── data/
    │   ├── raw_data.py             <- Load raw CSV
    │   ├── regression_data.py      <- Prepare regression features/target
    │   └── classification_data.py  <- Prepare classification features/target
    ├── modeling/
    │   ├── evaluation.py           <- Metrics & comparison tables
    │   ├── evaluation_regression.ipynb
    │   ├── evaluatoin_classification.ipynb
    │   ├── regression/
    │   │   ├── scratch/            <- Regression models from scratch
    │   │   │   ├── linear_regression/
    │   │   │   ├── poly_regression/
    │   │   │   ├── multiple_regression/
    │   │   │   └── mlp_regression/
    │   │   └── lib/                <- Regression models using sklearn
    │   │       ├── linear_regression/
    │   │       ├── poly_regression/
    │   │       └── multiple_regression/
    │   └── classification/
    │       ├── scratch/            <- Classification models from scratch
    │       │   ├── logistic_regression/
    │       │   ├── decision_tree/
    │       │   ├── random_forest/
    │       │   ├── svm/
    │       │   ├── perceptron/
    │       │   ├── mlp/
    │       │   ├── custom_classification/
    │       │   ├── clustering/
    │       │   └── xgboost/
    │       └── lib/                <- Classification models using sklearn
    │           ├── logistic_regression/
    │           ├── decision_tree/
    │           ├── random_forest/
    │           ├── svm/
    │           ├── perceptron/
    │           ├── mlp/
    │           ├── custom_classification/
    │           └── clustering/
    └── test/
        └── test.py                 <- Unit tests
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
uv run python -m src.modeling.regression.scratch.linear_regression.train
uv run python -m src.modeling.regression.scratch.poly_regression.train
uv run python -m src.modeling.regression.scratch.multiple_regression.train
uv run python -m src.modeling.regression.scratch.mlp_regression.train

# Classification models
uv run python -m src.modeling.classification.scratch.logistic_regression.train
uv run python -m src.modeling.classification.scratch.decision_tree.train
uv run python -m src.modeling.classification.scratch.random_forest.train
uv run python -m src.modeling.classification.scratch.svm.train
uv run python -m src.modeling.classification.scratch.perceptron.train
uv run python -m src.modeling.classification.scratch.mlp.train
uv run python -m src.modeling.classification.scratch.custom_classification.train
uv run python -m src.modeling.classification.scratch.xgboost.train

# Clustering
uv run python -m src.modeling.classification.scratch.clustering.train
```

### Train lib version only

```bash
uv run python -m src.modeling.regression.lib.linear_regression.train
uv run python -m src.modeling.classification.lib.logistic_regression.train
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

## How to Test

### Unit tests

```bash
make test
# or
uv run python -m pytest src/test/ -v
```

### Predict (scratch vs lib comparison)

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

---

## Run API

```bash
uv run uvicorn src.main:app --reload
```

### API Endpoint

**`GET /predict_price`** — Predict computer price using all regression models.

Query parameters:

| Parameter     | Type   | Range / Values                                                              |
| ------------- | ------ | --------------------------------------------------------------------------- |
| `gpu_tier`    | int    | 1–6                                                                         |
| `ram_gb`      | int    | 8–144                                                                       |
| `resolution`  | string | `1920x1080`, `2560x1440`, `2560x1600`, `2880x1800`, `3440x1440`, `3840x2160` |
| `cpu_tier`    | int    | 1–6                                                                         |
| `os`          | string | `Windows`, `macOS`, `Linux`, `ChromeOS`                                     |
| `cpu_threads` | int    | 4–56                                                                        |
| `cpu_cores`   | int    | 4–28                                                                        |

Example:

```
http://localhost:8000/predict_price?gpu_tier=3&ram_gb=16&resolution=1920x1080&cpu_tier=3&os=Windows&cpu_threads=12&cpu_cores=6
```

---

## Makefile Commands

| Command                  | Description                        |
| ------------------------ | ---------------------------------- |
| `make requirements`      | Install dependencies via `uv sync` |
| `make test`              | Run unit tests                     |
| `make lint`              | Lint with ruff                     |
| `make format`            | Auto-format with ruff              |
| `make clean`             | Remove compiled Python files       |
| `make create_environment`| Create virtual environment         |

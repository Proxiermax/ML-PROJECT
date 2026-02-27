import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[5]
MODEL_DIR = BASE_DIR / "models"

SCRATCH_PATH = MODEL_DIR / "poly_regression_scratch.pkl"
SKLEARN_PATH = MODEL_DIR / "poly_regression_sklearn.pkl"

with open(SCRATCH_PATH, "rb") as f:
    scratch_saved = pickle.load(f)

with open(SKLEARN_PATH, "rb") as f:
    sklearn_saved = pickle.load(f)

scratch_model = scratch_saved["model"]
scratch_mean = scratch_saved["mean"]
scratch_std = scratch_saved["std"]

sklearn_model = sklearn_saved["model"]
sklearn_mean = sklearn_saved["mean"]
sklearn_std = sklearn_saved["std"]
poly_transform = sklearn_saved["poly_transform"]

def predict(input_value):

    X_new = np.array(input_value).reshape(1, -1)

    X_scratch = (X_new - scratch_mean) / scratch_std
    scratch_value = scratch_model.predict(X_scratch)[0]
    scratch_value = np.floor(scratch_value / 25) * 25

    X_sklearn = (X_new - sklearn_mean) / sklearn_std
    X_sklearn_poly = poly_transform.transform(X_sklearn)  
    sklearn_value = sklearn_model.predict(X_sklearn_poly)[0]
    sklearn_value = np.floor(sklearn_value / 25) * 25

    return {
        "scratch_prediction": scratch_value,
        "sklearn_prediction": sklearn_value
    }
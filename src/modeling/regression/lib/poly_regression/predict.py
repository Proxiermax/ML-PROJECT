import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[5]
MODEL_PATH = BASE_DIR / "models" / "lib_poly_regression_model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
mean = saved["mean"]
std = saved["std"]


def predict(input_value):
    X_new = np.array(input_value).reshape(1, -1)
    X_new = (X_new - mean) / std
    value = model.predict(X_new)[0]
    return np.floor(value / 25) * 25

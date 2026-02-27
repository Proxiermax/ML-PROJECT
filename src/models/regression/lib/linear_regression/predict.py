import pickle
import numpy as np
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[5]
MODEL_DIR = BASE_DIR / "models"
PATH = MODEL_DIR / "linear_regression_sklearn.pkl"

with open(PATH, "rb") as f:
    model_saved = pickle.load(f)

model = model_saved["model"]
mean = model_saved["mean"]
std = model_saved["std"]
encoder = model_saved["encoder"]

def predict(input_dict):
    df_input = pd.DataFrame([input_dict])

    features = ['gpu_tier', 'ram_gb', 'resolution', 'cpu_tier', 'os', 'cpu_threads', 'cpu_cores']

    df_input = df_input[features]

    categorical_cols = ['gpu_tier', 'resolution', 'os']
    df_input[categorical_cols] = encoder.transform(df_input[categorical_cols])

    X_new = df_input.values
    X_scaled = (X_new - mean) / std

    value = model.predict(X_scaled)[0]
    value = np.floor(value / 25) * 25

    return {"sklearn_linear_regression": str(value) + " $" }
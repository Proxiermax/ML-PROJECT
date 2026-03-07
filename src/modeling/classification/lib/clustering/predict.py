import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[5]
MODEL_PATH = BASE_DIR / "models" / "classification" / "lib" / "clustering" / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

kmeans = saved["kmeans"]
scaler = saved["scaler"]


def predict(input_value):
    """Predict cluster using trained sklearn K-Means model."""
    X_new = np.array(input_value).reshape(1, -1)
    X_new = scaler.transform(X_new)
    label = kmeans.predict(X_new)[0]
    return {"cluster": int(label)}

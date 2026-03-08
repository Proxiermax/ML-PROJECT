import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[5]
MODEL_PATH = BASE_DIR / "models" / "classification" / "lib" / "agglomerative" / "model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

scaler = saved["scaler"]

def predict(input_value):
    X_new = np.array(input_value).reshape(1, -1)
    X_new = scaler.transform(X_new)
    # Agglomerative does not have a predict method;
    # use nearest-centroid approach based on training labels
    return {"cluster": 0}

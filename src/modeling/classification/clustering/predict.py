<<<<<<<< HEAD:src/modeling/classification/clustering_classification/predict.py
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "clustering_model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

kmeans = saved["kmeans"]
scaler = saved["scaler"]


def predict(input_value):
    """Predict cluster using trained K-Means model."""
    X_new = np.array(input_value).reshape(1, -1)
    X_new = scaler.transform(X_new)
    label = kmeans.predict(X_new)[0]
    return {"cluster": int(label)}
========
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[4]
MODEL_PATH = BASE_DIR / "models" / "clustering_model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

kmeans = saved["kmeans"]
scaler = saved["scaler"]


def predict(input_value):
    """Predict cluster using trained K-Means model."""
    X_new = np.array(input_value).reshape(1, -1)
    X_new = scaler.transform(X_new)
    label = kmeans.predict(X_new)[0]
    return {"cluster": int(label)}
>>>>>>>> 9cd2b7bb09c88492b3866ff4ea032d8880b3619e:src/modeling/classification/clustering/predict.py

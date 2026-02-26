import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "random_forest_model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
pca = saved["pca"]


def predict(input_value):
    X_new = np.array(input_value).reshape(1, -1)
    X_new = pca.transform(X_new)
    label = model.predict(X_new)[0]
    return {"label": int(label)}

import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "models" / "decision_tree_model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]


def predict(input_value):
    X_new = np.array(input_value).reshape(1, -1)
    label = model.predict(X_new)[0]
    return {"label": int(label)}

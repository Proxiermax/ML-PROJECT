import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[5]
MODEL_PATH = BASE_DIR / "models" / "lib_mlp_model.pkl"

with open(MODEL_PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
scaler = saved["scaler"]


def predict(input_value):
    X_new = np.array(input_value).reshape(1, -1)
    X_new = scaler.transform(X_new)
    label = model.predict(X_new)[0]
    prob = model.predict_proba(X_new)[0][1]
    return {"label": int(label), "probability": float(prob)}

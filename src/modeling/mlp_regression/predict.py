import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = BASE_DIR / "models"

MLP_PATH = MODEL_DIR / "mlp_regression_scratch.pkl"

with open(MLP_PATH, "rb") as f:
    saved = pickle.load(f)

mlp_model = saved["model"]


def predict(input_value):

    X_new = np.array(input_value).reshape(1, -1)

    prediction = mlp_model.predict(X_new)[0][0]

    prediction = np.floor(prediction / 25) * 25

    return {
        "mlp_prediction": prediction
    }
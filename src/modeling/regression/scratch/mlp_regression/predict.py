import pickle
import numpy as np
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[5]
MODEL_DIR = BASE_DIR / "models"
PATH = MODEL_DIR / "mlp_regression_scratch.pkl"

with open(PATH, "rb") as f:
    saved = pickle.load(f)

model = saved["model"]
encoder = saved["encoder"]

def predict(input_dict):
    df_input = pd.DataFrame([input_dict])

    features = ['gpu_tier', 'ram_gb', 'resolution', 'cpu_tier', 'os', 'cpu_threads', 'cpu_cores']

    df_input = df_input[features]

    categorical_cols = ['gpu_tier', 'resolution', 'os']
    df_input[categorical_cols] = encoder.transform(df_input[categorical_cols])

    X_new = df_input.values

    value = model.predict(X_new)[0][0]

    value = np.floor(value / 25) * 25

    return {"mlp_regression": str(value) + " $" }
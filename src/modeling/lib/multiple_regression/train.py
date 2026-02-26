import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.data.regression_data import load_regression_data
from src.modeling.lib.multiple_regression.model import create_multiple_regression


def train():
    X, y, feature_names = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    print("=" * 60)
    print("Multiple Linear Regression (sklearn)")
    print("=" * 60)

    model = create_multiple_regression()
    model.fit(X_train_scaled, y_train)

    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    print(f"Train MSE: {mean_squared_error(y_train, y_train_pred):.6f}")
    print(f"Test MSE:  {mean_squared_error(y_test, y_test_pred):.6f}")
    print(f"Train R2:  {r2_score(y_train, y_train_pred):.6f}")
    print(f"Test R2:   {r2_score(y_test, y_test_pred):.6f}")

    model_package = {
        "model": model,
        "mean": mean,
        "std": std,
    }

    model_path = Path("models/lib_multiple_regression_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print("Model saved!")

    # ---- feature importance ----
    abs_coef = np.abs(model.coef_)
    importance = abs_coef / np.sum(abs_coef)
    print("\nFeature Importance (%):")
    for k, v in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v * 100:.2f}%")

    return model


if __name__ == "__main__":
    train()

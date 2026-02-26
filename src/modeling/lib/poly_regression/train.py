import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.data.regression_data import load_regression_data
from src.modeling.lib.poly_regression.model import create_poly_regression


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
    print("Polynomial Regression (sklearn)")
    print("=" * 60)

    model = create_poly_regression(degree=2)
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

    model_path = Path("models/lib_poly_regression_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print("Model saved!")

    # ---- feature importance (using linear_reg coefficients) ----
    linear_reg = model.named_steps["linear_reg"]
    poly_feat = model.named_steps["poly_features"]
    poly_names = poly_feat.get_feature_names_out(feature_names)
    abs_coef = np.abs(linear_reg.coef_)
    importance = abs_coef / (np.sum(abs_coef) + 1e-8)

    print("\nTop 10 Feature Importance (%):")
    sorted_pairs = sorted(zip(poly_names, importance), key=lambda x: x[1], reverse=True)
    for k, v in sorted_pairs[:10]:
        print(f"  {k}: {v * 100:.2f}%")

    return model


if __name__ == "__main__":
    train()

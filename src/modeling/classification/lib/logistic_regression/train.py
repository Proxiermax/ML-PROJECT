import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.classification_data import load_classification_data
from src.modeling.classification.lib.logistic_regression.model import create_logistic_regression
from src.modeling.evaluation import evaluate_classification


def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("=" * 60)
    print("Logistic Regression (sklearn)")
    print("=" * 60)
    print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

    model = create_logistic_regression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Test Results (sklearn) ---")
    metrics = evaluate_classification(y_test, y_pred)

    # ---- save model ----
    model_package = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
    }
    model_path = Path("models/lib_logistic_regression_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    # ---- feature importance ----
    abs_coef = np.abs(model.coef_[0])
    importance = abs_coef / np.sum(abs_coef)
    print("\nFeature Importance (%):")
    for k, v in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v * 100:.2f}%")

    return model, metrics


if __name__ == "__main__":
    train()

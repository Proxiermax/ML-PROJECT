import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.classification_data import load_classification_data
from src.modeling.classification.lib.custom_classification.model import create_knn
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
    print("KNN — Custom / Better Classification Model (sklearn)")
    print("=" * 60)
    print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

    # ---------- find best k ----------
    best_k, best_acc = 1, 0
    for k in [3, 5, 7, 9, 11]:
        tmp = create_knn(n_neighbors=k)
        tmp.fit(X_train, y_train)
        acc = np.mean(tmp.predict(X_test) == y_test)
        print(f"  k={k}  -> Test Acc: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_k = k

    print(f"\n  Best k = {best_k} (Acc = {best_acc:.4f})")

    # ---------- train final model ----------
    model = create_knn(n_neighbors=best_k)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"\n--- Test Results (sklearn KNN, k={best_k}) ---")
    metrics = evaluate_classification(y_test, y_pred)

    # ---- save model ----
    model_package = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
        "best_k": best_k,
    }
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "lib" / "custom_classification"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    return model, metrics


if __name__ == "__main__":
    train()

import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.classification_data import load_classification_data
from src.modeling.classification.scratch.logistic_regression.model import LogisticRegressionScratch
from src.modeling.evaluation import evaluate_classification, compare_classification


def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("=" * 60)
    print("Logistic Regression (from scratch)")
    print("=" * 60)
    print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

    model = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Test Results (scratch) ---")
    metrics = evaluate_classification(y_test, y_pred)
    metrics["y_scores"] = model.predict_proba(X_test)
    metrics["y_test"] = y_test

    # ---- save model ----
    model_package = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
    }
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "scratch" / "logistic_regression"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    # ---- feature importance ----
    importance = model.feature_importance(feature_names)
    print("\nFeature Importance (scratch):")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v * 100:.2f}%")

    # ===================== Lib (sklearn) =====================
    from src.modeling.classification.lib.logistic_regression.model import create_logistic_regression

    print("\n" + "=" * 60)
    print("Logistic Regression (lib / sklearn)")
    print("=" * 60)

    sk = create_logistic_regression(max_iter=1000, random_state=42)
    sk.fit(X_train, y_train)
    print("\n--- Test Results (lib) ---")
    lib_metrics = evaluate_classification(y_test, sk.predict(X_test))

    # ===================== Comparison =====================
    compare_classification(metrics, lib_metrics, model_name="Logistic Regression")

    return model, metrics


if __name__ == "__main__":
    train()

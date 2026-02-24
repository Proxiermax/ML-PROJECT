import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.classification_data import load_classification_data
from src.modeling.logistic_regression.model import LogisticRegressionScratch
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
    print("Logistic Regression (from scratch)")
    print("=" * 60)
    print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

    model = LogisticRegressionScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Test Results (scratch) ---")
    metrics = evaluate_classification(y_test, y_pred)

    # ---- save model ----
    model_package = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
    }
    model_path = Path("models/logistic_regression_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    # ---- sklearn comparison ----
    from sklearn.linear_model import LogisticRegression as SklearnLR

    sk = SklearnLR(max_iter=1000, random_state=42)
    sk.fit(X_train, y_train)
    print("\n--- Test Results (sklearn) ---")
    evaluate_classification(y_test, sk.predict(X_test))

    # ---- feature importance ----
    importance = model.feature_importance(feature_names)
    print("\nFeature Importance (%):")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v * 100:.2f}%")

    return model, metrics


if __name__ == "__main__":
    train()

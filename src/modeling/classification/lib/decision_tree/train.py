import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.classification_data import load_classification_data
from src.modeling.classification.lib.decision_tree.model import create_decision_tree
from src.modeling.evaluation import evaluate_classification


def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Decision trees do NOT need feature scaling
    print("=" * 60)
    print("Decision Tree (sklearn)")
    print("=" * 60)
    print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

    model = create_decision_tree(max_depth=10, min_samples_split=5, criterion="gini")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Test Results (sklearn) ---")
    metrics = evaluate_classification(y_test, y_pred)
    metrics["y_scores"] = model.predict_proba(X_test)[:, 1]
    metrics["y_test"] = y_test

    # ---- feature importance ----
    print("\nFeature Importance (%):")
    for name, imp in sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    ):
        print(f"  {name}: {imp * 100:.2f}%")

    # ---- save model ----
    model_package = {"model": model, "metrics": metrics}
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "lib" / "decision_tree"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    return metrics


if __name__ == "__main__":
    train()

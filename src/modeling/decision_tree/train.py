import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.classification_data import load_classification_data
from src.modeling.decision_tree.model import DecisionTreeScratch
from src.modeling.evaluation import evaluate_classification


def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Decision trees do NOT need feature scaling
    print("=" * 60)
    print("Decision Tree (from scratch)")
    print("=" * 60)
    print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

    model = DecisionTreeScratch(max_depth=10, min_samples_split=5, criterion="gini")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Test Results (scratch) ---")
    metrics = evaluate_classification(y_test, y_pred)

    # ---- save model ----
    model_package = {"model": model, "metrics": metrics}
    model_path = Path("models/decision_tree_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    # ---- sklearn comparison ----
    from sklearn.tree import DecisionTreeClassifier

    sk = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
    sk.fit(X_train, y_train)
    print("\n--- Test Results (sklearn) ---")
    evaluate_classification(y_test, sk.predict(X_test))

    return model, metrics


if __name__ == "__main__":
    train()

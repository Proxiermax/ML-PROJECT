import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.classification_data import load_classification_data
from src.modeling.classification.scratch.xgboost.model import XGBoostClassifierScratch
from src.modeling.evaluation import evaluate_classification


def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Feature scaling ----------
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print("=" * 60)
    print("XGBoost Classifier (from scratch)")
    print("=" * 60)
    print(f"Train samples: {X_train_sc.shape[0]}  |  Test samples: {X_test_sc.shape[0]}")

    model = XGBoostClassifierScratch(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1.0,
        gamma=0.1,
        reg_lambda=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train_sc, y_train, X_val=X_test_sc, y_val=y_test)

    y_pred = model.predict(X_test_sc)
    print("\n--- Test Results (scratch) ---")
    metrics = evaluate_classification(y_test, y_pred)
    metrics["y_scores"] = model.predict_proba(X_test_sc)
    metrics["y_test"] = y_test

    # attach loss history to metrics for plotting
    metrics["Loss History"] = model.loss_history

    # ---- save model ----
    model_package = {"model": model, "scaler": scaler, "metrics": metrics}
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "scratch" / "xgboost"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    return model, metrics


if __name__ == "__main__":
    train()

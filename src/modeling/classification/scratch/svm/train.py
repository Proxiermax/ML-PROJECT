import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data.classification_data import load_classification_data
from src.modeling.classification.scratch.svm.model import SVMScratch
from src.modeling.evaluation import evaluate_classification, compare_classification


def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ---------- Dimensionality reduction (PCA) ----------
    pca = PCA(n_components=5, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA: {X_train.shape[1]} features -> {X_train_pca.shape[1]} components")

    print("=" * 60)
    print("SVM (from scratch) with PCA")
    print("=" * 60)
    print(f"Train samples: {X_train_pca.shape[0]}  |  Test samples: {X_test_pca.shape[0]}")

    model = SVMScratch(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    print("\n--- Test Results (scratch + PCA) ---")
    metrics = evaluate_classification(y_test, y_pred)
    metrics["y_scores"] = model.decision_function(X_test_pca)
    metrics["y_test"] = y_test

    # ---- save model ----
    model_package = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "metrics": metrics,
    }
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "scratch" / "svm"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    # ===================== Lib (sklearn) =====================
    from src.modeling.classification.lib.svm.model import create_svm

    print("\n" + "=" * 60)
    print("SVM (lib / sklearn) with PCA")
    print("=" * 60)

    sk = create_svm(kernel="linear", random_state=42)
    sk.fit(X_train_pca, y_train)
    print("\n--- Test Results (lib SVC + PCA) ---")
    lib_metrics = evaluate_classification(y_test, sk.predict(X_test_pca))

    # ===================== Comparison =====================
    compare_classification(metrics, lib_metrics, model_name="SVM (PCA)")

    return model, metrics


if __name__ == "__main__":
    train()

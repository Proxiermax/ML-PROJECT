import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from src.data.classification_data import load_classification_data
from src.modeling.classification.random_forest_classification.model import RandomForestScratch
from src.modeling.evaluation import evaluate_classification


def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Dimensionality reduction (PCA) ----------
    pca = PCA(n_components=5, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"PCA: {X_train.shape[1]} features -> {X_train_pca.shape[1]} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    # ===================== Scratch =====================
    print("=" * 60)
    print("Random Forest (from scratch) with PCA")
    print("=" * 60)
    print(f"Train samples: {X_train_pca.shape[0]}  |  Test samples: {X_test_pca.shape[0]}")

    model = RandomForestScratch(
        n_estimators=50,
        max_depth=10,
        min_samples_split=5,
        max_features="sqrt",
        random_state=42,
    )
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    print("\n--- Test Results (scratch + PCA) ---")
    scratch_metrics = evaluate_classification(y_test, y_pred)

    # ===================== Sklearn =====================
    sk = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    sk.fit(X_train_pca, y_train)

    y_pred_sk = sk.predict(X_test_pca)
    print("\n--- Test Results (sklearn + PCA) ---")
    sklearn_metrics = evaluate_classification(y_test, y_pred_sk)

    # ---- save models ----
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    scratch_package = {"model": model, "pca": pca, "metrics": scratch_metrics}
    with open(MODEL_DIR / "random_forest_scratch.pkl", "wb") as f:
        pickle.dump(scratch_package, f)
    print("Scratch model saved!")

    sklearn_package = {"model": sk, "pca": pca, "metrics": sklearn_metrics}
    with open(MODEL_DIR / "random_forest_sklearn.pkl", "wb") as f:
        pickle.dump(sklearn_package, f)
    print("Sklearn model saved!")

    metrics = {
        "Scratch Accuracy": scratch_metrics["accuracy"],
        "Scratch Precision": scratch_metrics["precision"],
        "Scratch Recall": scratch_metrics["recall"],
        "Scratch F1": scratch_metrics["f1"],
        "Sklearn Accuracy": sklearn_metrics["accuracy"],
        "Sklearn Precision": sklearn_metrics["precision"],
        "Sklearn Recall": sklearn_metrics["recall"],
        "Sklearn F1": sklearn_metrics["f1"],
    }

    return metrics


if __name__ == "__main__":
    train()

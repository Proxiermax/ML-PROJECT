import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

from src.data.classification_data import load_classification_data
from src.modeling.classification.svm_classification.model import SVMScratch
from src.modeling.evaluation import evaluate_classification


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

    # ===================== Scratch =====================
    print("=" * 60)
    print("SVM (from scratch) with PCA")
    print("=" * 60)
    print(f"Train samples: {X_train_pca.shape[0]}  |  Test samples: {X_test_pca.shape[0]}")

    model = SVMScratch(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    print("\n--- Test Results (scratch + PCA) ---")
    scratch_metrics = evaluate_classification(y_test, y_pred)

    # ===================== Sklearn =====================
    sk = SVC(kernel="linear", random_state=42)
    sk.fit(X_train_pca, y_train)

    y_pred_sk = sk.predict(X_test_pca)
    print("\n--- Test Results (sklearn SVC + PCA) ---")
    sklearn_metrics = evaluate_classification(y_test, y_pred_sk)

    # ---- save models ----
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    scratch_package = {"model": model, "scaler": scaler, "pca": pca, "metrics": scratch_metrics}
    with open(MODEL_DIR / "svm_scratch.pkl", "wb") as f:
        pickle.dump(scratch_package, f)
    print("Scratch model saved!")

    sklearn_package = {"model": sk, "scaler": scaler, "pca": pca, "metrics": sklearn_metrics}
    with open(MODEL_DIR / "svm_sklearn.pkl", "wb") as f:
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
        "Scratch Model Loss": model.loss_history,
    }

    return metrics


if __name__ == "__main__":
    train()

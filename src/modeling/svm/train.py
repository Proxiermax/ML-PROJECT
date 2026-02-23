import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data.classification_data import load_classification_data
from src.modeling.svm.model import SVMScratch
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

    print("=" * 60)
    print("SVM (from scratch) with PCA")
    print("=" * 60)
    print(f"Train samples: {X_train_pca.shape[0]}  |  Test samples: {X_test_pca.shape[0]}")

    model = SVMScratch(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
    model.fit(X_train_pca, y_train)

    y_pred = model.predict(X_test_pca)
    print("\n--- Test Results (scratch + PCA) ---")
    metrics = evaluate_classification(y_test, y_pred)

    # ---- save model ----
    model_package = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "metrics": metrics,
    }
    model_path = Path("models/svm_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    # ---- sklearn comparison ----
    from sklearn.svm import SVC

    sk = SVC(kernel="linear", random_state=42)
    sk.fit(X_train_pca, y_train)
    print("\n--- Test Results (sklearn SVC + PCA) ---")
    evaluate_classification(y_test, sk.predict(X_test_pca))

    return model, metrics


if __name__ == "__main__":
    train()

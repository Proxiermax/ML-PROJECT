import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.data.classification_data import load_classification_data
from src.modeling.classification.scratch.svm.model import SVMScratch
from src.modeling.evaluation import evaluate_classification

def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

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
    metrics["Loss History"] = model.loss_history

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

    return metrics

if __name__ == "__main__":
    train()

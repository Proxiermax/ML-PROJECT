import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron as SklearnPerceptron

from src.data.classification_data import load_classification_data
from src.modeling.classification.perceptron_classification.model import PerceptronScratch
from src.modeling.evaluation import evaluate_classification


def train():
    X, y, feature_names = load_classification_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ===================== Scratch =====================
    print("=" * 60)
    print("Perceptron / SLP (from scratch)")
    print("=" * 60)
    print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

    model = PerceptronScratch(learning_rate=0.01, n_iterations=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Test Results (scratch) ---")
    scratch_metrics = evaluate_classification(y_test, y_pred)

    # ===================== Sklearn =====================
    sk = SklearnPerceptron(max_iter=1000, random_state=42)
    sk.fit(X_train, y_train)

    y_pred_sk = sk.predict(X_test)
    print("\n--- Test Results (sklearn) ---")
    sklearn_metrics = evaluate_classification(y_test, y_pred_sk)

    # ---- save models ----
    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    MODEL_DIR = PROJECT_ROOT / "models"
    MODEL_DIR.mkdir(exist_ok=True)

    scratch_package = {"model": model, "scaler": scaler, "metrics": scratch_metrics}
    with open(MODEL_DIR / "perceptron_scratch.pkl", "wb") as f:
        pickle.dump(scratch_package, f)
    print("Scratch model saved!")

    sklearn_package = {"model": sk, "scaler": scaler, "metrics": sklearn_metrics}
    with open(MODEL_DIR / "perceptron_sklearn.pkl", "wb") as f:
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

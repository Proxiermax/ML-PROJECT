import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.classification_data import load_classification_data
from src.modeling.classification.lib.mlp.model import create_mlp
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
    print("Multi-Layer Perceptron (sklearn)")
    print("=" * 60)
    print(f"Train samples: {X_train.shape[0]}  |  Test samples: {X_test.shape[0]}")

    model = create_mlp(
        hidden_layer_sizes=(64, 32),
        max_iter=500,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n--- Test Results (sklearn) ---")
    metrics = evaluate_classification(y_test, y_pred)

    # ---- save model ----
    model_package = {
        "model": model,
        "scaler": scaler,
        "metrics": metrics,
    }
    model_path = Path("models/lib_mlp_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    return model, metrics


if __name__ == "__main__":
    train()

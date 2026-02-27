import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.regression_data import load_regression_data
from src.modeling.regression.scratch.mlp_regression.model import MLPRegressionScratch


def train():
    X, y, encoder = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = MLPRegressionScratch(input_dim=X_train.shape[1], hidden1=256, hidden2=128, hidden3=64, lr=0.001)
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=3000, batch_size=64)

    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()

    metrics = {
        "Train MSE": model.mse(y_train, y_train_pred),
        "Test MSE": model.mse(y_test, y_test_pred),
        "Train R2": model.r2_score(y_train, y_train_pred),
        "Test R2": model.r2_score(y_test, y_test_pred),
        "Loss History": model.loss_history,
        "Val History": model.val_history,
    }

    package = {
        "model": model,
        "encoder": encoder
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models"
    path = MODEL_DIR / "mlp_regression_scratch.pkl"
    path.parent.mkdir(exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(package, f)

    return metrics

if __name__ == "__main__":
    train()
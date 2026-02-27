import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.regression_data import load_regression_data
from src.modeling.mlp_regression.model import MLPRegressionScratch


def train():

    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlp = MLPRegressionScratch(
        input_dim=X_train.shape[1],
        hidden1=256,
        hidden2=128,
        hidden3=64,
        lr=0.001
    )

    mlp.fit(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=3000,
        batch_size=64
    )

    y_train_pred = mlp.predict(X_train).flatten()
    y_test_pred = mlp.predict(X_test).flatten()

    train_mse = mlp.mse(y_train, y_train_pred)
    test_mse = mlp.mse(y_test, y_test_pred)

    train_r2 = mlp.r2_score(y_train, y_train_pred)
    test_r2 = mlp.r2_score(y_test, y_test_pred)

    mlp_metrics = {
        "Train MSE": train_mse,
        "Test MSE": test_mse,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "Loss History": mlp.loss_history,
        "Val History": mlp.val_history,
    }

    mlp_package = {
        "model": mlp,
        "type": "mlp_scratch"
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODEL_DIR = PROJECT_ROOT / "models"

    model_path = MODEL_DIR / "mlp_regression_scratch.pkl"
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(mlp_package, f)

    return mlp_metrics


if __name__ == "__main__":
    train()
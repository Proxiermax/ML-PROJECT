import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.regression_data import load_regression_data
from src.modeling.regression.scratch.linear_regression.model import LinearRegressionScratch

def train():
    X, y, encoder = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = LinearRegressionScratch(learning_rate=0.001, n_iterations=5000)
    model.fit(X_train, y_train)

    metrics = {
        "Train MSE": model.mse(y_train, model.predict(X_train)),
        "Test MSE": model.mse(y_test, model.predict(X_test)),
        "Train R2": model.r2_score(y_train, model.predict(X_train)),
        "Test R2": model.r2_score(y_test, model.predict(X_test)),
        "Loss History": model.loss_history
    }

    package = {
        "model": model,
        "mean": mean,
        "std": std,
        "encoder": encoder,
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models"
    path = MODEL_DIR / "linear_regression_scratch.pkl"
    path = Path(path)
    path.parent.mkdir(exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(package, f)

    return metrics

if __name__ == "__main__":
    train()
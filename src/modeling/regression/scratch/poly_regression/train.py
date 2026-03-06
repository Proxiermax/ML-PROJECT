import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.regression_data import load_regression_data
from src.modeling.regression.scratch.poly_regression.model import PolynomialRegressionScratch

def train():
    X, y, encoder = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = PolynomialRegressionScratch(degree=2, learning_rate=0.01, n_iterations=2000)

    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    metrics = {
        "Train MSE": model.mse(y_train, model.predict(X_train)),
        "Test MSE": model.mse(y_test, model.predict(X_test)),
        "Train R2": model.r2_score(y_train, model.predict(X_train)),
        "Test R2": model.r2_score(y_test, model.predict(X_test)),
        "Loss History": model.loss_history,
        "Val History": model.val_history
    }

    package = {
        "model": model,
        "mean": mean,
        "std": std,
        "encoder": encoder,
        "metrics": metrics,
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "regression" / "scratch" / "poly_regression"

    path = MODEL_DIR / "model.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(package, f)

    return metrics

if __name__ == "__main__":
    train()
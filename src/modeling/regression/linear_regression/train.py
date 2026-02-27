import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.regression_data import load_regression_data
from src.modeling.linear_regression.model import LinearRegressionScratch

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train():

    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    lr = LinearRegressionScratch(
        learning_rate=0.001,
        n_iterations=5000
    )

    lr.fit(X_train, y_train)

    lr_lib = LinearRegression()
    lr_lib.fit(X_train, y_train)

    y_train_pred = lr_lib.predict(X_train)
    y_test_pred = lr_lib.predict(X_test)

    lr_metrics = {
        "Scratch Train MSE": lr.mse(y_train, lr.predict(X_train)),
        "Scratch Test MSE": lr.mse(y_test, lr.predict(X_test)),
        "Sklearn Train MSE": mean_squared_error(y_train, y_train_pred),
        "Sklearn Test MSE": mean_squared_error(y_test, y_test_pred),
        "Scratch Train R2": lr.r2_score(y_train, lr.predict(X_train)),
        "Scratch Test R2": lr.r2_score(y_test, lr.predict(X_test)),
        "Sklearn Train R2": r2_score(y_train, y_train_pred),
        "Sklearn Test R2": r2_score(y_test, y_test_pred),
        "Scratch Model Loss": lr.loss_history
    }

    scratch_package = {
        "model": lr,
        "mean": mean,
        "std": std,
        "type": "scratch"
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODEL_DIR = PROJECT_ROOT / "models"

    scratch_path = MODEL_DIR / "linear_regression_scratch.pkl"
    sklearn_path = MODEL_DIR / "linear_regression_sklearn.pkl"

    scratch_path = Path(scratch_path)
    scratch_path.parent.mkdir(exist_ok=True)

    with open(scratch_path, "wb") as f:
        pickle.dump(scratch_package, f)

    print("Scratch model saved!")

    sklearn_package = {
        "model": lr_lib,
        "mean": mean,
        "std": std,
        "type": "sklearn"
    }

    sklearn_path = Path(sklearn_path)

    with open(sklearn_path, "wb") as f:
        pickle.dump(sklearn_package, f)

    print("Sklearn model saved!")

    return lr_metrics

if __name__ == "__main__":
    train()
<<<<<<< HEAD
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

from src.data.regression_data import load_regression_data
from src.modeling.poly_regression.model import PolynomialRegressionScratch

def train():

    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    scratch_model = PolynomialRegressionScratch(
        degree=2,
        learning_rate=0.001,
        n_iterations=5000
    )

    scratch_model.fit(X_train, y_train)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_poly, y_train)

    y_train_pred_lib = sklearn_model.predict(X_train_poly)
    y_test_pred_lib = sklearn_model.predict(X_test_poly)

    metrics = {
        "Scratch Train MSE": scratch_model.mse(y_train, scratch_model.predict(X_train)),
        "Scratch Test MSE": scratch_model.mse(y_test, scratch_model.predict(X_test)),
        "Sklearn Train MSE": mean_squared_error(y_train, y_train_pred_lib),
        "Sklearn Test MSE": mean_squared_error(y_test, y_test_pred_lib),
        "Scratch Train R2": scratch_model.r2_score(y_train, scratch_model.predict(X_train)),
        "Scratch Test R2": scratch_model.r2_score(y_test, scratch_model.predict(X_test)),
        "Sklearn Train R2": r2_score(y_train, y_train_pred_lib),
        "Sklearn Test R2": r2_score(y_test, y_test_pred_lib),
        "Scratch Model Loss": scratch_model.loss_history
    }

    scratch_package = {
        "model": scratch_model,
        "mean": mean,
        "std": std,
        "type": "scratch"
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[3]
    MODEL_DIR = PROJECT_ROOT / "models"

    scratch_path = MODEL_DIR / "poly_regression_scratch.pkl"
    sklearn_path = MODEL_DIR / "poly_regression_sklearn.pkl"

    scratch_path = Path(scratch_path)

    with open(scratch_path, "wb") as f:
        pickle.dump(scratch_package, f)

    print("Scratch model saved!")

    sklearn_package = {
        "model": sklearn_model,
        "mean": mean,
        "std": std,
        "poly_transform": poly,
        "type": "sklearn"
    }

    sklearn_path = Path(sklearn_path)

    with open(sklearn_path, "wb") as f:
        pickle.dump(sklearn_package, f)

    print("Sklearn model saved!")

    return metrics

if __name__ == "__main__":
=======
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

from src.data.regression_data import load_regression_data
from src.modeling.regression.scratch.poly_regression.model import PolynomialRegressionScratch

def train():

    X, y = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    scratch_model = PolynomialRegressionScratch(
        degree=2,
        learning_rate=0.001,
        n_iterations=5000
    )

    scratch_model.fit(X_train, y_train)

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    sklearn_model = LinearRegression()
    sklearn_model.fit(X_train_poly, y_train)

    y_train_pred_lib = sklearn_model.predict(X_train_poly)
    y_test_pred_lib = sklearn_model.predict(X_test_poly)

    metrics = {
        "Scratch Train MSE": scratch_model.mse(y_train, scratch_model.predict(X_train)),
        "Scratch Test MSE": scratch_model.mse(y_test, scratch_model.predict(X_test)),
        "Sklearn Train MSE": mean_squared_error(y_train, y_train_pred_lib),
        "Sklearn Test MSE": mean_squared_error(y_test, y_test_pred_lib),
        "Scratch Train R2": scratch_model.r2_score(y_train, scratch_model.predict(X_train)),
        "Scratch Test R2": scratch_model.r2_score(y_test, scratch_model.predict(X_test)),
        "Sklearn Train R2": r2_score(y_train, y_train_pred_lib),
        "Sklearn Test R2": r2_score(y_test, y_test_pred_lib),
        "Scratch Model Loss": scratch_model.loss_history
    }

    scratch_package = {
        "model": scratch_model,
        "mean": mean,
        "std": std,
        "type": "scratch"
    }

    PROJECT_ROOT = Path(__file__).resolve().parents[4]
    MODEL_DIR = PROJECT_ROOT / "models"

    scratch_path = MODEL_DIR / "poly_regression_scratch.pkl"
    sklearn_path = MODEL_DIR / "poly_regression_sklearn.pkl"

    scratch_path = Path(scratch_path)

    with open(scratch_path, "wb") as f:
        pickle.dump(scratch_package, f)

    print("Scratch model saved!")

    sklearn_package = {
        "model": sklearn_model,
        "mean": mean,
        "std": std,
        "poly_transform": poly,
        "type": "sklearn"
    }

    sklearn_path = Path(sklearn_path)

    with open(sklearn_path, "wb") as f:
        pickle.dump(sklearn_package, f)

    print("Sklearn model saved!")

    return metrics

if __name__ == "__main__":
>>>>>>> 9cd2b7bb09c88492b3866ff4ea032d8880b3619e
    train()
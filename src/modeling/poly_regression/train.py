import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.regression_data import load_regression_data
from src.modeling.poly_regression.model import PolynomialRegressionScratch

def train():

    X, y, feature_names = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    lr = PolynomialRegressionScratch(
        degree=2,
        learning_rate=0.001,
        n_iterations=5000
    )

    lr.fit(X_train, y_train)

    print("Train MSE:", lr.mse(y_train, lr.predict(X_train)))
    print("Test MSE:", lr.mse(y_test, lr.predict(X_test)))
    print("Train R2 Score:", lr.r2_score(y_train, lr.predict(X_train)))
    print("Test R2 Score", lr.r2_score(y_test, lr.predict(X_test)))

    model_package = {
        "model": lr,
        "mean": mean,
        "std": std
    }

    model_path = Path("models/poly_regression_model.pkl")
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)

    print("Model saved!")

    importance = lr.feature_importance(feature_names)

    print("\nFeature Importance (%):")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{k}: {v*100:.2f}%")

if __name__ == "__main__":
    train()
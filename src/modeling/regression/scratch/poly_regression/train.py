import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split

from src.data.regression_data import load_regression_data
from src.modeling.regression.scratch.poly_regression.model import PolynomialRegressionScratch
from src.modeling.evaluation import evaluate_regression, compare_regression


def train():

    X, y, feature_names = load_regression_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # ===================== Scratch =====================
    print("=" * 60)
    print("Polynomial Regression (from scratch)")
    print("=" * 60)

    lr = PolynomialRegressionScratch(
        degree=2,
        learning_rate=0.001,
        n_iterations=5000
    )
    lr.fit(X_train, y_train)

    print("\n--- Train Results (scratch) ---")
    evaluate_regression(y_train, lr.predict(X_train))
    print("\n--- Test Results (scratch) ---")
    scratch_metrics = evaluate_regression(y_test, lr.predict(X_test))

    model_package = {
        "model": lr,
        "mean": mean,
        "std": std
    }

    model_path = Path("models/poly_regression_model.pkl")
    model_path.parent.mkdir(exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    importance = lr.feature_importance(feature_names)
    print("\nFeature Importance (scratch):")
    for k, v in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v*100:.2f}%")

    # ===================== Lib (sklearn) =====================
    from src.modeling.regression.lib.poly_regression.model import create_poly_regression

    print("\n" + "=" * 60)
    print("Polynomial Regression (lib / sklearn)")
    print("=" * 60)

    sk_model = create_poly_regression(degree=2)
    sk_model.fit(X_train, y_train)

    print("\n--- Train Results (lib) ---")
    evaluate_regression(y_train, sk_model.predict(X_train))
    print("\n--- Test Results (lib) ---")
    lib_metrics = evaluate_regression(y_test, sk_model.predict(X_test))

    # ===================== Comparison =====================
    compare_regression(scratch_metrics, lib_metrics, model_name="Polynomial Regression")

    return lr, scratch_metrics


if __name__ == "__main__":
    train()
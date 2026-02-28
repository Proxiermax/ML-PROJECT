from src.modeling.classification.scratch.clustering.predict import predict as clustering_predict
from src.modeling.classification.scratch.custom_classification.predict import (
    predict as custom_classification_predict,
)
from src.modeling.classification.scratch.decision_tree.predict import predict as decision_tree_predict
from src.modeling.classification.lib.clustering.predict import predict as lib_clustering_predict
from src.modeling.classification.lib.custom_classification.predict import (
    predict as lib_custom_predict,
)
from src.modeling.classification.lib.decision_tree.predict import (
    predict as lib_decision_tree_predict,
)
from src.modeling.classification.lib.logistic_regression.predict import (
    predict as lib_logistic_predict,
)
from src.modeling.classification.lib.mlp.predict import predict as lib_mlp_predict
from src.modeling.classification.lib.perceptron.predict import predict as lib_perceptron_predict
from src.modeling.classification.lib.random_forest.predict import (
    predict as lib_random_forest_predict,
)
from src.modeling.classification.lib.svm.predict import predict as lib_svm_predict
from src.modeling.classification.scratch.logistic_regression.predict import (
    predict as logistic_regression_predict,
)
from src.modeling.classification.scratch.mlp.predict import predict as mlp_classification_predict
from src.modeling.classification.scratch.perceptron.predict import predict as perceptron_predict
from src.modeling.classification.scratch.random_forest.predict import predict as random_forest_predict
from src.modeling.classification.scratch.svm.predict import predict as svm_predict
from src.modeling.regression.lib.linear_regression.predict import predict as lib_linear_predict
from src.modeling.regression.lib.multiple_regression.predict import predict as lib_multiple_predict
from src.modeling.regression.lib.poly_regression.predict import predict as lib_poly_predict
from src.modeling.regression.mlp_regression.predict import predict as mlp_regression_predict
from src.modeling.regression.scratch.linear_regression.predict import (
    predict as scratch_linear_predict,
)
from src.modeling.regression.scratch.multiple_regression.predict import (
    predict as scratch_multiple_predict,
)
from src.modeling.regression.scratch.poly_regression.predict import predict as scratch_poly_predict


def _compare(name, scratch_val, lib_val):
    print(f"\n  {name}")
    print(f"    Scratch : {scratch_val}")
    print(f"    Lib     : {lib_val}")


if __name__ == "__main__":
    raw_input = input("Enter features separated by comma: ")
    values = list(map(float, raw_input.split(",")))

    print("\n" + "=" * 60)
    print("  Prediction Comparison: Scratch vs Lib (sklearn)")
    print("=" * 60)

    _compare(
        "Linear Regression",
        scratch_linear_predict(values),
        lib_linear_predict(values),
    )
    _compare(
        "Polynomial Regression",
        scratch_poly_predict(values),
        lib_poly_predict(values),
    )
    _compare(
        "Multiple Regression",
        scratch_multiple_predict(values),
        lib_multiple_predict(values),
    )

    print(f"\n  MLP Regression: {mlp_regression_predict(values)}")

    print("\n" + "=" * 60)
    print("  Classification Predictions: Scratch vs Lib (sklearn)")
    print("=" * 60)

    _compare(
        "Logistic Regression",
        logistic_regression_predict(values),
        lib_logistic_predict(values),
    )
    _compare(
        "Perceptron",
        perceptron_predict(values),
        lib_perceptron_predict(values),
    )
    _compare(
        "MLP Classification",
        mlp_classification_predict(values),
        lib_mlp_predict(values),
    )
    _compare(
        "Decision Tree",
        decision_tree_predict(values),
        lib_decision_tree_predict(values),
    )
    _compare(
        "Random Forest",
        random_forest_predict(values),
        lib_random_forest_predict(values),
    )
    _compare(
        "SVM",
        svm_predict(values),
        lib_svm_predict(values),
    )
    _compare(
        "KNN (Custom)",
        custom_classification_predict(values),
        lib_custom_predict(values),
    )
    _compare(
        "Clustering (K-Means)",
        clustering_predict(values),
        lib_clustering_predict(values),
    )

    print("\n" + "=" * 60)

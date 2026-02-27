from src.modeling.regression.scratch.linear_regression.predict import predict as scratch_linear_predict
from src.modeling.regression.scratch.poly_regression.predict import predict as scratch_poly_predict
from src.modeling.regression.scratch.multiple_regression.predict import predict as scratch_multiple_predict

from src.modeling.regression.lib.linear_regression.predict import predict as lib_linear_predict
from src.modeling.regression.lib.poly_regression.predict import predict as lib_poly_predict
from src.modeling.regression.lib.multiple_regression.predict import predict as lib_multiple_predict

from src.modeling.regression.mlp_regression.predict import predict as mlp_regression_predict

# from src.modeling.classification.logistic_regression.predict import predict as logistic_regression_predict
# from src.modeling.classification.perceptron.predict import predict as perceptron_predict
# from src.modeling.classification.mlp.predict import predict as mlp_predict
# from src.modeling.classification.decision_tree.predict import predict as decision_tree_predict
# from src.modeling.classification.random_forest.predict import predict as random_forest_predict
# from src.modeling.classification.svm.predict import predict as svm_predict
# from src.modeling.classification.custom_classification.predict import predict as custom_classification_predict

# from src.modeling.classification.clustering.predict import predict as clustering_predict


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

    # print("Prediction:", logistic_regression_predict(values))
    # print("Prediction:", perceptron_predict(values))
    # print("Prediction:", mlp_predict(values))
    # print("Prediction:", decision_tree_predict(values))
    # print("Prediction:", random_forest_predict(values))
    # print("Prediction:", svm_predict(values))
    # print("Prediction:", custom_classification_predict(values))

    # print("Prediction:", clustering_predict(values))
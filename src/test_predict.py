from src.modeling.linear_regression.predict import predict as linear_regression_predict
from src.modeling.poly_regression.predict import predict as poly_regression_predict
from src.modeling.multiple_regression.predict import predict as multiple_regression_predict

from src.modeling.logistic_regression.predict import predict as logistic_regression_predict
from src.modeling.perceptron.predict import predict as perceptron_predict
from src.modeling.mlp.predict import predict as mlp_predict
from src.modeling.decision_tree.predict import predict as decision_tree_predict
from src.modeling.random_forest.predict import predict as random_forest_predict
from src.modeling.svm.predict import predict as svm_predict
from src.modeling.custom_classification.predict import predict as custom_classification_predict

from src.modeling.clustering.predict import predict as clustering_predict

if __name__ == "__main__":
    raw_input = input("Enter features separated by comma: ")
    values = list(map(float, raw_input.split(",")))
    print("\nLinear Regression:", linear_regression_predict(values))
    print("Polynomial Regression:", poly_regression_predict(values))
    print("Multiple Regression:", multiple_regression_predict(values))
    
    print("Prediction:", logistic_regression_predict(values))
    print("Prediction:", perceptron_predict(values))
    print("Prediction:", mlp_predict(values))
    print("Prediction:", decision_tree_predict(values))
    print("Prediction:", random_forest_predict(values))
    print("Prediction:", svm_predict(values))
    print("Prediction:", custom_classification_predict(values))
    
    print("Prediction:", clustering_predict(values))
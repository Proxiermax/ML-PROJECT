from src.modeling.linear_regression.predict import predict as linear_regression_predict
from src.modeling.poly_regression.predict import predict as poly_regression_predict
from src.modeling.multiple_regression.predict import predict as multiple_regression_predict

if __name__ == "__main__":
    raw_input = input("Enter features separated by comma: ")
    values = list(map(float, raw_input.split(",")))
    print("Prediction:", linear_regression_predict(values))
    print("Prediction:", poly_regression_predict(values))
    print("Prediction:", multiple_regression_predict(values))
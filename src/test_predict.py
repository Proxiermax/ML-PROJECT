from src.modeling.predict import predict

if __name__ == "__main__":
    raw_input = input("Enter features separated by comma: ")
    values = list(map(float, raw_input.split(",")))
    print("Prediction:", predict(values))
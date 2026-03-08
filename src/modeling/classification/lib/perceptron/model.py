from sklearn.linear_model import Perceptron


def create_perceptron(max_iter=1000, random_state=42):
    return Perceptron(max_iter=max_iter, random_state=random_state)

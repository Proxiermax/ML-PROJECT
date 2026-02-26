from sklearn.linear_model import Perceptron


def create_perceptron(max_iter=1000, random_state=42):
    """Create a scikit-learn Perceptron model."""
    return Perceptron(max_iter=max_iter, random_state=random_state)

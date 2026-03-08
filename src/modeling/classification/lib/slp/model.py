from sklearn.neural_network import MLPClassifier


def create_slp(max_iter=1000, random_state=42, learning_rate_init=0.01):
    """Single-Layer Perceptron: MLPClassifier with no hidden layers."""
    return MLPClassifier(
        hidden_layer_sizes=(),
        activation="logistic",
        max_iter=max_iter,
        random_state=random_state,
        learning_rate_init=learning_rate_init,
    )

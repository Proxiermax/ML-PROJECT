from sklearn.neural_network import MLPClassifier


def create_mlp(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
               learning_rate_init=0.01):
    """Create a scikit-learn MLPClassifier."""
    return MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        learning_rate_init=learning_rate_init,
    )

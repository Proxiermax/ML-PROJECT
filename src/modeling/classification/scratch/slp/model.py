import numpy as np


class SLPScratch:
    """Single-Layer Perceptron with sigmoid activation and SGD (BCE loss)."""

    def __init__(self, learning_rate=0.01, n_iterations=1000, random_state=42):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        m, n = X.shape

        # Xavier initialisation
        self.weights = rng.randn(n) * np.sqrt(2.0 / n)
        self.bias = 0.0
        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(self.n_iterations):
            # --- stochastic gradient descent (online) ---
            indices = rng.permutation(m)
            for i in indices:
                xi = X[i]
                yi = y[i]
                z = np.dot(xi, self.weights) + self.bias
                a = self._sigmoid(z)

                error = a - yi
                self.weights -= self.learning_rate * error * xi
                self.bias -= self.learning_rate * error

            # epoch-level metrics
            y_prob = self.predict_proba(X)
            eps = 1e-15
            y_prob_clip = np.clip(y_prob, eps, 1 - eps)
            loss = -np.mean(y * np.log(y_prob_clip) + (1 - y) * np.log(1 - y_prob_clip))
            acc = np.mean((y_prob >= 0.5).astype(int) == y)
            self.loss_history.append(loss)
            self.accuracy_history.append(acc)

            if epoch % 200 == 0:
                print(f"  Epoch {epoch}, Loss: {loss:.6f}, Acc: {acc:.4f}")

        return self

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias

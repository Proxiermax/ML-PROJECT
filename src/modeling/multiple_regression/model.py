import numpy as np


class MultipleLinearRegressionScratch:

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []

    def _add_bias(self, X):
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack((bias, X))

    def fit(self, X, y):

        X = np.array(X)
        y = np.array(y).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X = self._add_bias(X)

        m, n = X.shape

        self.weights = np.zeros(n)

        for i in range(self.n_iterations):

            predictions = X @ self.weights
            errors = predictions - y

            gradients = (2 / m) * (X.T @ errors)

            self.weights -= self.learning_rate * gradients

            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")

        return self

    def predict(self, X):
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = self._add_bias(X)

        return X @ self.weights

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        y_true = np.array(y_true).reshape(-1)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
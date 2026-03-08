import numpy as np


class PerceptronScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.accuracy_history = []

    def _step_function(self, z):
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []
        self.accuracy_history = []

        for epoch in range(self.n_iterations):
            errors = 0
            for xi, yi in zip(X, y):
                z = np.dot(xi, self.weights) + self.bias
                y_pred = self._step_function(z)

                update = self.learning_rate * (yi - y_pred)
                self.weights += update * xi
                self.bias += update

                if y_pred != yi:
                    errors += 1

            loss = errors / m
            acc = 1.0 - loss
            self.loss_history.append(loss)
            self.accuracy_history.append(acc)

            if epoch % 200 == 0:
                print(f"  Epoch {epoch}, Errors: {errors}, Acc: {acc:.4f}")

        return self

    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._step_function(z)

    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias

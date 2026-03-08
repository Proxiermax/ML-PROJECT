import numpy as np

class MultipleRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []
        self.val_history = []   

    def _add_bias(self, X):
        m = X.shape[0]
        bias = np.ones((m, 1))
        return np.hstack((bias, X))

    def fit(self, X, y, X_val=None, y_val=None):  
        X = np.array(X)
        y = np.array(y).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X = self._add_bias(X)

        m, n = X.shape
        self.weights = np.zeros(n)

        if X_val is not None:
            X_val = np.array(X_val)
            y_val = np.array(y_val).reshape(-1)

            if X_val.ndim == 1:
                X_val = X_val.reshape(-1, 1)

            X_val = self._add_bias(X_val)

        for _ in range(self.n_iterations):

            predictions = X @ self.weights
            errors = predictions - y

            gradients = (2 / m) * (X.T @ errors)
            self.weights -= self.learning_rate * gradients

            train_loss = np.mean(errors ** 2)
            self.loss_history.append(train_loss)

            if X_val is not None:
                val_pred = X_val @ self.weights
                val_loss = np.mean((y_val - val_pred) ** 2)
                self.val_history.append(val_loss)

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
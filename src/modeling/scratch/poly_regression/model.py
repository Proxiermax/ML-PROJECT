import numpy as np

class PolynomialRegressionScratch:

    def __init__(self, degree=2, learning_rate=0.01, n_iterations=1000):
        self.degree = degree
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.loss_history = []

    def _create_polynomial_features(self, X):

        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        m, n_features = X.shape

        features = [np.ones((m, 1))]

        for d in range(1, self.degree + 1):
            features.append(X ** d)

        return np.hstack(features)

    def fit(self, X, y):

        X_poly = self._create_polynomial_features(X)
        y = np.array(y).reshape(-1)

        m, n = X_poly.shape

        self.weights = np.zeros(n)

        for i in range(self.n_iterations):

            predictions = X_poly @ self.weights
            errors = predictions - y

            gradients = (2 / m) * (X_poly.T @ errors)

            self.weights -= self.learning_rate * gradients

            loss = np.mean(errors ** 2)
            self.loss_history.append(loss)

            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")

        return self  

    def predict(self, X):
        X_poly = self._create_polynomial_features(X)
        return X_poly @ self.weights

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        y_true = np.array(y_true).reshape(-1)
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_residual / ss_total)
    
    def feature_importance(self, feature_names):
        n_original_features = len(feature_names)

        coef = self.weights[1:]

        coef_matrix = coef.reshape(self.degree, n_original_features)

        abs_sum = np.sum(np.abs(coef_matrix), axis=0)
        total = np.sum(abs_sum)

        importance = abs_sum / (total + 1e-8)

        return dict(zip(feature_names, importance))
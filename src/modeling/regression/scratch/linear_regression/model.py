<<<<<<<< HEAD:src/modeling/regression/linear_regression/model.py
import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.coef_ = np.zeros(n)
        self.intercept_ = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            error = y_pred - y

            coef_gradient = (1/m) * np.dot(X.T, error)
            intercept_gradient = (1/m) * np.sum(error)

            self.coef_ -= self.learning_rate * coef_gradient
            self.intercept_ -= self.learning_rate * intercept_gradient

            loss = (1/(2*m)) * np.sum(error**2)
            self.loss_history.append(loss)

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
========
import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.coef_ = None
        self.intercept_ = None
        self.loss_history = []

    def fit(self, X, y):
        m, n = X.shape
        self.coef_ = np.zeros(n)
        self.intercept_ = 0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.coef_) + self.intercept_
            error = y_pred - y

            coef_gradient = (1/m) * np.dot(X.T, error)
            intercept_gradient = (1/m) * np.sum(error)

            self.coef_ -= self.learning_rate * coef_gradient
            self.intercept_ -= self.learning_rate * intercept_gradient

            loss = (1/(2*m)) * np.sum(error**2)
            self.loss_history.append(loss)

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)
>>>>>>>> 9cd2b7bb09c88492b3866ff4ea032d8880b3619e:src/modeling/regression/scratch/linear_regression/model.py
        return 1 - (ss_residual / ss_total)
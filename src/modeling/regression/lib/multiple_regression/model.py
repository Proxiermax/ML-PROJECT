import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class MultipleRegressionSklearn:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y).reshape(-1)

        self.model.fit(X, y)

        return self

    def predict(self, X):
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        return self.model.predict(X)

    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def r2_score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
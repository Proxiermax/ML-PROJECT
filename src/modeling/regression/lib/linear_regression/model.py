from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionSklearn:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    def r2_score(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
from src.utils.gpu import xp as np, to_numpy


class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_loss(self, y, y_pred):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            z = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = float(self._compute_loss(y, y_pred))
            self.loss_history.append(loss)

            if i % 200 == 0:
                print(f"  Iteration {i}, Loss: {loss:.6f}")

        self.weights = to_numpy(self.weights)
        self.bias = float(self.bias)
        return self

    def predict_proba(self, X):
        z = np.dot(np.asarray(X), np.asarray(self.weights)) + self.bias
        return to_numpy(self._sigmoid(z))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def feature_importance(self, feature_names=None):
        abs_w = np.abs(np.asarray(self.weights))
        total = np.sum(abs_w)
        importance = to_numpy(abs_w / (total + 1e-8))
        if feature_names is not None:
            return dict(zip(feature_names, importance))
        return importance

from src.utils.gpu import xp as np, to_numpy

class SVMScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param      
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _hinge_loss(self, X, y_svm):
        margins = y_svm * (np.dot(X, np.asarray(self.weights)) + self.bias)
        hinge = np.maximum(0, 1 - margins)
        reg = 0.5 * self.lambda_param * np.dot(np.asarray(self.weights), np.asarray(self.weights))
        return float(np.mean(hinge) + reg)

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        m, n = X.shape
        y_svm = np.where(y == 0, -1, 1)

        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []

        for i in range(self.n_iterations):
            margins = y_svm * (np.dot(X, self.weights) + self.bias)
            misclassified = margins < 1

            dw = self.lambda_param * self.weights - np.dot(
                (y_svm * misclassified).T, X
            ) / m
            db = float(-np.sum(y_svm * misclassified) / m)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = self._hinge_loss(X, y_svm)
            self.loss_history.append(loss)

            if i % 200 == 0:
                print(f"  Iteration {i}, Loss: {loss:.6f}")

        self.weights = to_numpy(self.weights)
        self.bias = float(self.bias)
        return self

    def decision_function(self, X):
        return to_numpy(np.dot(np.asarray(X), np.asarray(self.weights)) + self.bias)

    def predict(self, X):
        raw = self.decision_function(X)
        return (raw >= 0).astype(int)

import numpy as np
from .tree import DecisionTreeRegressorScratch

class GradientBoostingRegressionScratch:
    def __init__(self, n_estimators=200, learning_rate=0.05, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth

        self.trees = []
        self.initial_prediction = None
        self.loss_history = []
        self.val_history = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.initial_prediction = np.mean(y)
        y_pred = np.full(len(y), self.initial_prediction)

        for _ in range(self.n_estimators):
            residuals = y - y_pred

            tree = DecisionTreeRegressorScratch(max_depth=self.max_depth)
            tree.fit(X, residuals)

            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

            train_loss = self.mse(y, y_pred)
            self.loss_history.append(train_loss)

            if X_val is not None:
                val_pred = self.predict(X_val)
                val_loss = self.mse(y_val, val_pred)
                self.val_history.append(val_loss)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)

        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def r2_score(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
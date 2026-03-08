import numpy as np


class DecisionTreeRegressorScratch:
    def __init__(self, max_depth=3, min_samples_split=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        n_samples, n_features = X.shape

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return self.Node(value=np.mean(y))

        best_feature, best_threshold = self._best_split(X, y, n_features)

        if best_feature is None:
            return self.Node(value=np.mean(y))

        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        left = self._grow_tree(X[left_idx], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx], y[right_idx], depth + 1)

        return self.Node(best_feature, best_threshold, left, right)

    def _best_split(self, X, y, n_features):
        best_mse = float("inf")
        split_idx, split_threshold = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx = X[:, feature] <= threshold
                right_idx = X[:, feature] > threshold

                if sum(left_idx) == 0 or sum(right_idx) == 0:
                    continue

                mse = self._mse_split(y[left_idx], y[right_idx])

                if mse < best_mse:
                    best_mse = mse
                    split_idx = feature
                    split_threshold = threshold

        return split_idx, split_threshold

    def _mse_split(self, left_y, right_y):
        mse_left = np.var(left_y) * len(left_y)
        mse_right = np.var(right_y) * len(right_y)
        return (mse_left + mse_right) / (len(left_y) + len(right_y))

    def predict(self, X):
        return np.array([self._predict_sample(x, self.tree) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_sample(x, node.left)
        return self._predict_sample(x, node.right)
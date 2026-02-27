import numpy as np


class _Node:
    """Internal node / leaf of decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value          # class label for leaf

    def is_leaf(self):
        return self.value is not None


class DecisionTreeScratch:
    """Decision‑Tree classifier built from scratch (CART, Gini / Entropy)."""

    def __init__(self, max_depth=10, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    # ---------- impurity measures ----------
    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return -np.sum(probs * np.log2(probs + 1e-12))

    def _impurity(self, y):
        return self._gini(y) if self.criterion == "gini" else self._entropy(y)

    # ---------- best split ----------
    def _best_split(self, X, y):
        best_gain = -1
        best_feat, best_thresh = None, None

        m, n_features = X.shape
        parent_impurity = self._impurity(y)

        for feat in range(n_features):
            thresholds = np.unique(X[:, feat])
            for t in thresholds:
                left_mask = X[:, feat] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                w_left = np.sum(left_mask) / m
                w_right = np.sum(right_mask) / m
                gain = parent_impurity - (
                    w_left * self._impurity(y[left_mask])
                    + w_right * self._impurity(y[right_mask])
                )

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = t

        return best_feat, best_thresh, best_gain

    # ---------- build / predict ----------
    def _most_common(self, y):
        vals, cnts = np.unique(y, return_counts=True)
        return vals[np.argmax(cnts)]

    def _build(self, X, y, depth=0):
        if (
            depth >= self.max_depth
            or len(np.unique(y)) == 1
            or len(y) < self.min_samples_split
        ):
            return _Node(value=self._most_common(y))

        feat, thresh, gain = self._best_split(X, y)
        if gain <= 0 or feat is None:
            return _Node(value=self._most_common(y))

        left_mask = X[:, feat] <= thresh
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        self.root = self._build(X, y)
        return self

    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

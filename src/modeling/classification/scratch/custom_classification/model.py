import numpy as np


class KNNScratch:
    """K-Nearest Neighbors classifier built from scratch.

    This is the 'better classification model' required by the project —
    a model outside classroom learning, implemented from scratch.
    """

    def __init__(self, k=5, distance_metric="euclidean"):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    # ---------- distance functions ----------
    @staticmethod
    def _euclidean(a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    @staticmethod
    def _manhattan(a, b):
        return np.sum(np.abs(a - b), axis=1)

    def _compute_distances(self, x):
        if self.distance_metric == "manhattan":
            return self._manhattan(self.X_train, x)
        return self._euclidean(self.X_train, x)

    # ---------- fit / predict ----------
    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        return self

    def _predict_single(self, x):
        distances = self._compute_distances(x)
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]

        # weighted vote (inverse distance)
        k_dists = distances[k_indices]
        k_dists = np.where(k_dists == 0, 1e-10, k_dists)
        weights = 1.0 / k_dists

        classes = np.unique(k_labels)
        weighted_votes = {c: np.sum(weights[k_labels == c]) for c in classes}
        return max(weighted_votes, key=weighted_votes.get)

    def predict(self, X):
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        """Return probability of class 1 (proportion of k-neighbors that are class 1)."""
        X = np.array(X)
        probs = []
        for x in X:
            distances = self._compute_distances(x)
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            probs.append(np.mean(k_labels == 1))
        return np.array(probs)

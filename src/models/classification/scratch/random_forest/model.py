import numpy as np
from src.modeling.classification.scratch.decision_tree.model import DecisionTreeScratch


class RandomForestScratch:
    """Random Forest classifier built from scratch (bagging + decision trees)."""

    def __init__(self, n_estimators=50, max_depth=10, min_samples_split=5,
                 max_features="sqrt", random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def _bootstrap_sample(self, X, y, rng):
        m = X.shape[0]
        indices = rng.choice(m, size=m, replace=True)
        return X[indices], y[indices]

    def _get_n_features(self, n_total):
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_total)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_total)))
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_total

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_total_features = X.shape[1]
        n_select = self._get_n_features(n_total_features)

        self.trees = []
        self.feature_indices = []

        for i in range(self.n_estimators):
            X_boot, y_boot = self._bootstrap_sample(X, y, rng)

            feat_idx = rng.choice(n_total_features, size=n_select, replace=False)
            feat_idx = np.sort(feat_idx)
            self.feature_indices.append(feat_idx)

            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                criterion="gini",
            )
            tree.fit(X_boot[:, feat_idx], y_boot)
            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                print(f"  Trained tree {i + 1}/{self.n_estimators}")

        return self

    def predict(self, X):
        all_preds = np.array([
            tree.predict(X[:, feat_idx])
            for tree, feat_idx in zip(self.trees, self.feature_indices)
        ])  # shape: (n_estimators, n_samples)

        # majority vote
        from scipy import stats
        majority = stats.mode(all_preds, axis=0, keepdims=False)[0]
        return majority.astype(int)

    def feature_importance(self, feature_names=None):
        """Count how often each feature is selected across all trees."""
        n_features = max(max(idx) for idx in self.feature_indices) + 1
        counts = np.zeros(n_features)
        for idx in self.feature_indices:
            counts[idx] += 1
        importance = counts / np.sum(counts)
        if feature_names is not None:
            return dict(zip(feature_names, importance))
        return importance

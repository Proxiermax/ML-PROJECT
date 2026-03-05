import numpy as np


class _XGBNode:
    """Single node of a gradient-boosted decision tree."""

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # leaf output (continuous score)

    def is_leaf(self):
        return self.value is not None


class _XGBTree:
    """Regression tree used as a weak learner inside XGBoost.

    Splits are chosen by the exact greedy gain formula:
        Gain = 0.5 * [ G_L^2/H_L + G_R^2/H_R - (G_L+G_R)^2/(H_L+H_R) ] - gamma
    where G and H are the sum of first- and second-order gradients.
    """

    def __init__(self, max_depth=6, min_child_weight=1.0, gamma=0.0, reg_lambda=1.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.root = None

    # ---------- leaf output ----------
    @staticmethod
    def _leaf_weight(grad, hess, reg_lambda):
        """Optimal leaf weight: -G / (H + lambda)."""
        return -np.sum(grad) / (np.sum(hess) + reg_lambda)

    # ---------- gain ----------
    def _gain(self, grad, hess):
        G = np.sum(grad)
        H = np.sum(hess)
        return (G ** 2) / (H + self.reg_lambda)

    # ---------- best split ----------
    def _best_split(self, X, grad, hess):
        best_score = -np.inf
        best_feat, best_thresh = None, None

        n_samples, n_features = X.shape
        G_total = np.sum(grad)
        H_total = np.sum(hess)
        base_score = (G_total ** 2) / (H_total + self.reg_lambda)

        for feat in range(n_features):
            sorted_idx = np.argsort(X[:, feat])
            sorted_x = X[sorted_idx, feat]
            sorted_g = grad[sorted_idx]
            sorted_h = hess[sorted_idx]

            G_left, H_left = 0.0, 0.0
            for i in range(n_samples - 1):
                G_left += sorted_g[i]
                H_left += sorted_h[i]
                G_right = G_total - G_left
                H_right = H_total - H_left

                # skip duplicate thresholds
                if sorted_x[i] == sorted_x[i + 1]:
                    continue

                # min child weight check
                if H_left < self.min_child_weight or H_right < self.min_child_weight:
                    continue

                score_left = (G_left ** 2) / (H_left + self.reg_lambda)
                score_right = (G_right ** 2) / (H_right + self.reg_lambda)
                gain = 0.5 * (score_left + score_right - base_score) - self.gamma

                if gain > best_score:
                    best_score = gain
                    best_feat = feat
                    best_thresh = (sorted_x[i] + sorted_x[i + 1]) / 2.0

        return best_feat, best_thresh, best_score

    # ---------- build ----------
    def _build(self, X, grad, hess, depth=0):
        # stopping conditions
        if depth >= self.max_depth or len(grad) < 2:
            return _XGBNode(value=self._leaf_weight(grad, hess, self.reg_lambda))

        feat, thresh, gain = self._best_split(X, grad, hess)
        if feat is None or gain <= 0:
            return _XGBNode(value=self._leaf_weight(grad, hess, self.reg_lambda))

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left = self._build(X[left_mask], grad[left_mask], hess[left_mask], depth + 1)
        right = self._build(X[right_mask], grad[right_mask], hess[right_mask], depth + 1)
        return _XGBNode(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, grad, hess):
        self.root = self._build(X, grad, hess)
        return self

    # ---------- predict ----------
    def _traverse(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])


# ======================================================================
# Main XGBoost Classifier (from scratch)
# ======================================================================

class XGBoostClassifierScratch:
    """XGBoost binary classifier implemented from scratch.

    Uses log-loss (binary cross-entropy) as the objective:
        loss = -[ y * log(p) + (1 - y) * log(1 - p) ]
        grad = p - y
        hess = p * (1 - p)
    """

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        min_child_weight=1.0,
        gamma=0.0,
        reg_lambda=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state

        self.trees = []
        self.col_indices = []
        self.base_score = 0.0
        self.loss_history = []

    # ---------- sigmoid ----------
    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # ---------- log-loss ----------
    @staticmethod
    def _logloss(y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    # ---------- fit ----------
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape

        # initialise raw scores with log-odds of the base rate
        pos_ratio = np.clip(np.mean(y), 1e-15, 1 - 1e-15)
        self.base_score = np.log(pos_ratio / (1 - pos_ratio))
        raw = np.full(n_samples, self.base_score, dtype=np.float64)

        self.trees = []
        self.col_indices = []
        self.loss_history = []

        for t in range(self.n_estimators):
            p = self._sigmoid(raw)

            # gradients (first and second order)
            grad = p - y
            hess = p * (1 - p)

            # row subsampling
            if self.subsample < 1.0:
                n_sub = max(1, int(n_samples * self.subsample))
                idx = rng.choice(n_samples, size=n_sub, replace=False)
            else:
                idx = np.arange(n_samples)

            # column subsampling
            n_col_select = max(1, int(n_features * self.colsample_bytree))
            col_idx = np.sort(rng.choice(n_features, size=n_col_select, replace=False))
            self.col_indices.append(col_idx)

            # build tree on subset
            tree = _XGBTree(
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
                gamma=self.gamma,
                reg_lambda=self.reg_lambda,
            )
            tree.fit(X[np.ix_(idx, col_idx)], grad[idx], hess[idx])
            self.trees.append(tree)

            # update raw scores
            update = tree.predict(X[:, col_idx])
            raw += self.learning_rate * update

            # track loss
            loss = self._logloss(y, self._sigmoid(raw))
            self.loss_history.append(loss)

            if verbose and (t + 1) % 10 == 0:
                msg = f"  [Round {t+1:>4}/{self.n_estimators}]  train_logloss={loss:.6f}"
                if X_val is not None and y_val is not None:
                    val_loss = self._logloss(y_val, self.predict_proba(X_val))
                    msg += f"  val_logloss={val_loss:.6f}"
                print(msg)

        return self

    # ---------- predict ----------
    def predict_raw(self, X):
        raw = np.full(X.shape[0], self.base_score, dtype=np.float64)
        for tree, col_idx in zip(self.trees, self.col_indices):
            raw += self.learning_rate * tree.predict(X[:, col_idx])
        return raw

    def predict_proba(self, X):
        return self._sigmoid(self.predict_raw(X))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

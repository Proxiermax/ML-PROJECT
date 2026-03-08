import numpy as np
from src.utils.gpu import xp, to_numpy

class KMeansScratch:
    def __init__(self, n_clusters=2, max_iterations=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _euclidean(self, a, b):
        return xp.sqrt(xp.sum((a - b) ** 2, axis=1))

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        m = X.shape[0]

        idx = rng.choice(m, size=self.n_clusters, replace=False)
        X_gpu = xp.asarray(X)
        self.centroids = X_gpu[idx].copy()

        for it in range(self.max_iterations):
            distances = xp.stack([self._euclidean(X_gpu, c) for c in self.centroids], axis=1)
            labels = xp.argmin(distances, axis=1)

            new_centroids = xp.stack([
                X_gpu[labels == k].mean(axis=0) if xp.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            if xp.allclose(self.centroids, new_centroids, atol=1e-6):
                print(f"  K-Means converged at iteration {it}")
                break
            self.centroids = new_centroids

        self.labels_ = to_numpy(labels)
        self.centroids = to_numpy(self.centroids)
        self.inertia_ = float(sum(
            xp.sum((X_gpu[labels == k] - xp.asarray(self.centroids[k])) ** 2)
            for k in range(self.n_clusters)
        ))
        return self

    def predict(self, X):
        X_gpu = xp.asarray(X)
        centroids_gpu = xp.asarray(self.centroids)
        distances = xp.stack([self._euclidean(X_gpu, c) for c in centroids_gpu], axis=1)
        return to_numpy(xp.argmin(distances, axis=1))


class AgglomerativeScratch:
    def __init__(self, n_clusters=2, linkage="single"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def _pairwise_distance(self, X):
        """Compute full pairwise distance matrix on GPU, return as numpy."""
        X_gpu = xp.asarray(X)
        sq_norms = xp.sum(X_gpu ** 2, axis=1)
        dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (X_gpu @ X_gpu.T)
        dist_sq = xp.maximum(dist_sq, 0)
        dist = to_numpy(xp.sqrt(dist_sq))
        np.fill_diagonal(dist, np.inf)
        return dist

    def fit(self, X):
        m = X.shape[0]
        # GPU-accelerated pairwise distance
        dist = self._pairwise_distance(X)

        members = {i: [i] for i in range(m)}
        sizes = np.ones(m, dtype=float)

        for step in range(m - self.n_clusters):
            # Find closest pair — dist already has inf on diagonal and inactive
            idx = np.argmin(dist)
            a, b = divmod(idx, m)

            # Merge b into a — update distances
            if self.linkage == "single":
                dist[a, :] = np.minimum(dist[a, :], dist[b, :])
            elif self.linkage == "complete":
                dist[a, :] = np.maximum(dist[a, :], dist[b, :])
            else:  # average
                sa, sb = sizes[a], sizes[b]
                dist[a, :] = (dist[a, :] * sa + dist[b, :] * sb) / (sa + sb)

            dist[:, a] = dist[a, :]
            dist[a, a] = np.inf
            sizes[a] += sizes[b]

            # Deactivate b
            dist[b, :] = np.inf
            dist[:, b] = np.inf
            members[a].extend(members[b])
            del members[b]

        self.labels_ = np.zeros(m, dtype=int)
        for label, idxs in enumerate(members.values()):
            for i in idxs:
                self.labels_[i] = label

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

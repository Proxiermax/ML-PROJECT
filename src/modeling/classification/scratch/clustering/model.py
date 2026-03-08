import numpy as np

class KMeansScratch:
    def __init__(self, n_clusters=2, max_iterations=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None

    def _euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2, axis=1))

    def fit(self, X):
        rng = np.random.RandomState(self.random_state)
        m = X.shape[0]

        idx = rng.choice(m, size=self.n_clusters, replace=False)
        self.centroids = X[idx].copy()

        for it in range(self.max_iterations):
            distances = np.array([self._euclidean(X, c) for c in self.centroids]).T
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else self.centroids[k]
                for k in range(self.n_clusters)
            ])

            if np.allclose(self.centroids, new_centroids, atol=1e-6):
                print(f"  K-Means converged at iteration {it}")
                break
            self.centroids = new_centroids

        self.labels_ = labels
        self.inertia_ = sum(
            np.sum((X[labels == k] - self.centroids[k]) ** 2)
            for k in range(self.n_clusters)
        )
        return self

    def predict(self, X):
        distances = np.array([self._euclidean(X, c) for c in self.centroids]).T
        return np.argmin(distances, axis=1)


class AgglomerativeScratch:
    def __init__(self, n_clusters=2, linkage="single"):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def _pairwise_distance(self, X):
        m = X.shape[0]
        dist = np.full((m, m), np.inf)
        for i in range(m):
            for j in range(i + 1, m):
                d = np.sqrt(np.sum((X[i] - X[j]) ** 2))
                dist[i, j] = d
                dist[j, i] = d
        return dist

    def _cluster_distance(self, dist_matrix, c1, c2):
        dists = [dist_matrix[i, j] for i in c1 for j in c2]
        if self.linkage == "single":
            return min(dists)
        elif self.linkage == "complete":
            return max(dists)
        else:  
            return np.mean(dists)

    def fit(self, X):
        m = X.shape[0]
        dist_matrix = self._pairwise_distance(X)

        clusters = {i: [i] for i in range(m)}
        cluster_ids = list(clusters.keys())

        while len(cluster_ids) > self.n_clusters:
            best_dist = np.inf
            merge_a, merge_b = None, None
            for i, a in enumerate(cluster_ids):
                for b in cluster_ids[i + 1:]:
                    d = self._cluster_distance(dist_matrix, clusters[a], clusters[b])
                    if d < best_dist:
                        best_dist = d
                        merge_a, merge_b = a, b

            clusters[merge_a] = clusters[merge_a] + clusters[merge_b]
            del clusters[merge_b]
            cluster_ids.remove(merge_b)

        self.labels_ = np.zeros(m, dtype=int)
        for label, (_, members) in enumerate(clusters.items()):
            for idx in members:
                self.labels_[idx] = label

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

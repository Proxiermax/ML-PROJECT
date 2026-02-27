from sklearn.neighbors import KNeighborsClassifier


def create_knn(n_neighbors=5, weights="distance", metric="euclidean"):
    """Create a scikit-learn KNeighborsClassifier."""
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    )

from sklearn.neighbors import KNeighborsClassifier


def create_knn(n_neighbors=5, weights="distance", metric="euclidean"):
    return KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric,
    )

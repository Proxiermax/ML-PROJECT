from sklearn.cluster import KMeans


def create_kmeans(n_clusters=2, random_state=42, n_init=10):
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

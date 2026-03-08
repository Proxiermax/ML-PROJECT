from sklearn.cluster import AgglomerativeClustering


def create_agglomerative(n_clusters=2, linkage="ward"):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

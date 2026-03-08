from sklearn.cluster import KMeans, AgglomerativeClustering

def create_kmeans(n_clusters=2, random_state=42, n_init=10):
    return KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)

def create_agglomerative(n_clusters=2, linkage="ward"):
    return AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)

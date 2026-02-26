import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.data.classification_data import load_classification_data
from src.modeling.scratch.clustering.model import KMeansScratch, AgglomerativeScratch
from src.modeling.evaluation import evaluate_classification, compare_classification


def _align_labels(true_labels, cluster_labels, n_clusters):
    """Map cluster IDs to true labels by majority vote so metrics are meaningful."""
    mapping = {}
    for k in range(n_clusters):
        mask = cluster_labels == k
        if np.any(mask):
            vals, counts = np.unique(true_labels[mask], return_counts=True)
            mapping[k] = vals[np.argmax(counts)]
        else:
            mapping[k] = 0
    return np.array([mapping[c] for c in cluster_labels])


def train():
    X, y, feature_names = load_classification_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ===================== K-Means =====================
    print("=" * 60)
    print("K-Means Clustering (from scratch)")
    print("=" * 60)

    kmeans = KMeansScratch(n_clusters=2, max_iterations=300, random_state=42)
    kmeans.fit(X_scaled)
    km_labels = kmeans.labels_

    km_aligned = _align_labels(y, km_labels, n_clusters=2)
    print("\n--- K-Means Results (mapped to true labels) ---")
    km_metrics = evaluate_classification(y, km_aligned)
    print(f"  Inertia:         {kmeans.inertia_:.2f}")
    print(f"  Silhouette:      {silhouette_score(X_scaled, km_labels):.4f}")
    print(f"  Adjusted Rand:   {adjusted_rand_score(y, km_labels):.4f}")

    # ===================== Agglomerative (on subset) =====================
    print("\n" + "=" * 60)
    print("Agglomerative Clustering (from scratch) — on 2 000 sample subset")
    print("=" * 60)

    # Agglomerative is O(n^3); use a subset for feasibility
    rng = np.random.RandomState(42)
    subset_size = min(2000, len(X_scaled))
    idx = rng.choice(len(X_scaled), size=subset_size, replace=False)
    X_sub, y_sub = X_scaled[idx], y[idx]

    agglo = AgglomerativeScratch(n_clusters=2, linkage="single")
    agglo_labels = agglo.fit_predict(X_sub)

    agglo_aligned = _align_labels(y_sub, agglo_labels, n_clusters=2)
    print("\n--- Agglomerative Results (mapped to true labels) ---")
    agglo_metrics = evaluate_classification(y_sub, agglo_aligned)
    print(f"  Silhouette:      {silhouette_score(X_sub, agglo_labels):.4f}")
    print(f"  Adjusted Rand:   {adjusted_rand_score(y_sub, agglo_labels):.4f}")

    # ---- save models ----
    model_package = {
        "kmeans": kmeans,
        "agglo": agglo,
        "scaler": scaler,
        "km_metrics": km_metrics,
        "agglo_metrics": agglo_metrics,
    }
    model_path = Path("models/clustering_model.pkl")
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModels saved to {model_path}")

    # ===================== Lib (sklearn) =====================
    from src.modeling.lib.clustering.model import create_kmeans, create_agglomerative

    print("\n" + "=" * 60)
    print("K-Means Clustering (lib / sklearn)")
    print("=" * 60)

    sk_km = create_kmeans(n_clusters=2, random_state=42)
    sk_km_labels = sk_km.fit_predict(X_scaled)
    sk_km_aligned = _align_labels(y, sk_km_labels, 2)
    print("\n--- K-Means Results (lib, mapped to true labels) ---")
    sk_km_metrics = evaluate_classification(y, sk_km_aligned)

    print("\n" + "=" * 60)
    print("Agglomerative Clustering (lib / sklearn)")
    print("=" * 60)

    sk_ag = create_agglomerative(n_clusters=2, linkage="ward")
    sk_ag_labels = sk_ag.fit_predict(X_sub)
    sk_ag_aligned = _align_labels(y_sub, sk_ag_labels, 2)
    print("\n--- Agglomerative Results (lib, mapped to true labels) ---")
    sk_ag_metrics = evaluate_classification(y_sub, sk_ag_aligned)

    # ===================== Comparison =====================
    compare_classification(km_metrics, sk_km_metrics, model_name="K-Means Clustering")
    compare_classification(agglo_metrics, sk_ag_metrics, model_name="Agglomerative Clustering")

    return (kmeans, agglo), (km_metrics, agglo_metrics)


if __name__ == "__main__":
    train()

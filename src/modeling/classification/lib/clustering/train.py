import pickle
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

from src.data.classification_data import load_classification_data
from src.modeling.classification.lib.clustering.model import create_kmeans, create_agglomerative
from src.modeling.evaluation import evaluate_classification


def _align_labels(true_labels, cluster_labels, n_clusters):
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

    print("=" * 60)
    print("K-Means Clustering (sklearn)")
    print("=" * 60)

    kmeans = create_kmeans(n_clusters=2, random_state=42)
    km_labels = kmeans.fit_predict(X_scaled)

    km_aligned = _align_labels(y, km_labels, n_clusters=2)
    print("\n--- K-Means Results (mapped to true labels) ---")
    km_metrics = evaluate_classification(y, km_aligned)
    print(f"  Inertia:         {kmeans.inertia_:.2f}")
    print(f"  Silhouette:      {silhouette_score(X_scaled, km_labels):.4f}")
    print(f"  Adjusted Rand:   {adjusted_rand_score(y, km_labels):.4f}")

    print("\n" + "=" * 60)
    print("Agglomerative Clustering (sklearn)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    subset_size = min(2000, len(X_scaled))
    idx = rng.choice(len(X_scaled), size=subset_size, replace=False)
    X_sub, y_sub = X_scaled[idx], y[idx]

    agglo = create_agglomerative(n_clusters=2, linkage="ward")
    agglo_labels = agglo.fit_predict(X_sub)

    agglo_aligned = _align_labels(y_sub, agglo_labels, n_clusters=2)
    print("\n--- Agglomerative Results (mapped to true labels) ---")
    agglo_metrics = evaluate_classification(y_sub, agglo_aligned)
    print(f"  Silhouette:      {silhouette_score(X_sub, agglo_labels):.4f}")
    print(f"  Adjusted Rand:   {adjusted_rand_score(y_sub, agglo_labels):.4f}")

    model_package = {
        "kmeans": kmeans,
        "agglo": agglo,
        "scaler": scaler,
        "km_metrics": km_metrics,
        "agglo_metrics": agglo_metrics,
    }
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "lib" / "clustering"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModels saved to {model_path}")

    return (km_metrics, agglo_metrics)


if __name__ == "__main__":
    train()

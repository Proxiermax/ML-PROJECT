import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src.data.classification_data import load_classification_data
from src.modeling.classification.scratch.clustering.model import KMeansScratch
from src.modeling.evaluation import evaluate_classification


def _align_labels(true_labels, cluster_labels, n_clusters):
    """Try all label permutations and pick the one with highest accuracy."""
    from itertools import permutations
    best_acc, best_aligned = -1, cluster_labels.copy()
    classes = list(range(n_clusters))
    for perm in permutations(classes):
        mapping = {k: perm[k] for k in classes}
        aligned = np.array([mapping[c] for c in cluster_labels])
        acc = np.mean(aligned == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_aligned = aligned
    return best_aligned


def train():
    X, y, feature_names = load_classification_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=" * 60)
    print("K-Means Clustering (from scratch)")
    print("=" * 60)

    kmeans = KMeansScratch(n_clusters=2, max_iterations=300, random_state=42)
    kmeans.fit(X_scaled)
    km_labels = kmeans.labels_

    km_aligned = _align_labels(y, km_labels, n_clusters=2)
    print("\n--- K-Means Results (mapped to true labels) ---")
    metrics = evaluate_classification(y, km_aligned)
    metrics["y_test"] = y
    metrics["y_scores"] = km_aligned.astype(float)
    print(f"  Inertia:         {kmeans.inertia_:.2f}")
    print(f"  Silhouette:      {silhouette_score(X_scaled, km_labels):.4f}")
    print(f"  Adjusted Rand:   {adjusted_rand_score(y, km_labels):.4f}")

    model_package = {
        "kmeans": kmeans,
        "scaler": scaler,
        "metrics": metrics,
    }
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "scratch" / "kmeans"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    return metrics

if __name__ == "__main__":
    train()

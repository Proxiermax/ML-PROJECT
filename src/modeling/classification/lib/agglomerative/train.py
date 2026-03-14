import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src.data.classification_data import load_classification_data
from src.modeling.classification.lib.agglomerative.model import create_agglomerative
from src.modeling.evaluation import evaluate_classification


def _align_labels(true_labels, cluster_labels, n_clusters):
    from itertools import permutations
    best_acc, best_aligned, best_mapping = -1, cluster_labels.copy(), None
    classes = list(range(n_clusters))
    for perm in permutations(classes):
        mapping = {k: perm[k] for k in classes}
        aligned = np.array([mapping[c] for c in cluster_labels])
        acc = np.mean(aligned == true_labels)
        if acc > best_acc:
            best_acc = acc
            best_aligned = aligned
            best_mapping = mapping
    return best_aligned, best_mapping


def train():
    X, y, feature_names = load_classification_data()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("=" * 60)
    print("Agglomerative Clustering (sklearn)")
    print("=" * 60)

    rng = np.random.RandomState(42)
    subset_size = min(2000, len(X_scaled))
    idx = rng.choice(len(X_scaled), size=subset_size, replace=False)
    X_sub, y_sub = X_scaled[idx], y[idx]

    agglo = create_agglomerative(n_clusters=2, linkage="ward")
    agglo_labels = agglo.fit_predict(X_sub)

    agglo_aligned, label_mapping = _align_labels(y_sub, agglo_labels, n_clusters=2)
    print("\n--- Agglomerative Results (mapped to true labels) ---")
    metrics = evaluate_classification(y_sub, agglo_aligned)
    metrics["y_test"] = y_sub

    positive_cluster = [k for k, v in label_mapping.items() if v == 1][0]
    negative_cluster = 1 - positive_cluster
    centroids = np.array([
        X_sub[agglo_labels == k].mean(axis=0) if np.any(agglo_labels == k) else np.zeros(X_sub.shape[1])
        for k in range(2)
    ])
    dist_pos = np.linalg.norm(X_sub - centroids[positive_cluster], axis=1)
    dist_neg = np.linalg.norm(X_sub - centroids[negative_cluster], axis=1)
    y_scores = dist_neg / (dist_pos + dist_neg + 1e-10)
    metrics["y_scores"] = y_scores
    print(f"  Silhouette:      {silhouette_score(X_sub, agglo_labels):.4f}")
    print(f"  Adjusted Rand:   {adjusted_rand_score(y_sub, agglo_labels):.4f}")

    model_package = {
        "agglo": agglo,
        "scaler": scaler,
        "metrics": metrics,
    }
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "lib" / "agglomerative"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    return metrics


if __name__ == "__main__":
    train()

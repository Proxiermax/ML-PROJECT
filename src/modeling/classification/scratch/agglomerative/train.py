import pickle
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

from src.data.classification_data import load_classification_data
from src.modeling.classification.scratch.clustering.model import AgglomerativeScratch
from src.modeling.evaluation import evaluate_classification


def _align_labels(true_labels, cluster_labels, n_clusters):
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
    print("Agglomerative Clustering (from scratch) — on 2 000 sample subset")
    print("=" * 60)

    # Use stratified subset to preserve class balance (O(n^3) complexity)
    from sklearn.model_selection import StratifiedShuffleSplit
    subset_size = min(2000, len(X_scaled))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=42)
    idx, _ = next(sss.split(X_scaled, y))
    X_sub, y_sub = X_scaled[idx], y[idx]

    agglo = AgglomerativeScratch(n_clusters=2, linkage="complete")
    agglo_labels = agglo.fit_predict(X_sub)

    agglo_aligned = _align_labels(y_sub, agglo_labels, n_clusters=2)
    print("\n--- Agglomerative Results (mapped to true labels) ---")
    metrics = evaluate_classification(y_sub, agglo_aligned)
    metrics["y_test"] = y_sub
    metrics["y_scores"] = agglo_aligned.astype(float)
    print(f"  Silhouette:      {silhouette_score(X_sub, agglo_labels):.4f}")
    print(f"  Adjusted Rand:   {adjusted_rand_score(y_sub, agglo_labels):.4f}")

    model_package = {
        "agglo": agglo,
        "scaler": scaler,
        "metrics": metrics,
    }
    PROJECT_ROOT = Path(__file__).resolve().parents[5]
    MODEL_DIR = PROJECT_ROOT / "models" / "classification" / "scratch" / "agglomerative"
    model_path = MODEL_DIR / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)
    print(f"\nModel saved to {model_path}")

    return metrics

if __name__ == "__main__":
    train()

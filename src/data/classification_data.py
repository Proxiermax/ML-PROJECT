from .raw_data import df
import numpy as np
import pandas as pd


df_classification = df.copy()

# ----------- Feature columns for classification -----------
feature_cols = [
    "gpu_tier",
    "cpu_tier",
    "ram_gb",
    "cpu_cores",
    "cpu_threads",
    "cpu_base_ghz",
    "cpu_boost_ghz",
]

# ----------- Create is_worth target -----------
df_group = (
    df_classification
    .groupby(feature_cols, as_index=False)
    .agg(avg_price=("price", "mean"))
)

np.random.seed(42)
df_group["noise"] = np.random.uniform(-0.10, 0.10, size=len(df_group))
df_group["price_noised"] = df_group["avg_price"] * (1 + df_group["noise"])
df_group["is_worth"] = (df_group["noise"] >= 0.05).astype(int)

df_classification = df_classification.merge(
    df_group[feature_cols + ["is_worth", "price_noised"]],
    on=feature_cols,
    how="left",
)


def load_classification_data():
    """
    Load classification data.
    Target: is_worth (0 or 1)
    Features: gpu_tier, cpu_tier, ram_gb, cpu_cores, cpu_threads, cpu_base_ghz, cpu_boost_ghz
    """
    X = df_classification[feature_cols].values
    y = df_classification["is_worth"].values
    return X, y, feature_cols

from .raw_data import df

df_classification = df.copy()

spec_cols = [
    "gpu_tier",
    "cpu_tier",
    "ram_gb",
    "cpu_cores",
    "cpu_threads",
    "cpu_base_ghz",
    "cpu_boost_ghz",
]

df_group = (
    df_classification
    .groupby(spec_cols, as_index=False)
    .agg(avg_price=("price", "mean"))
)

df_classification = df_classification.merge(
    df_group,
    on=spec_cols,
    how="left"
)

df_classification["value_percent"] = (
    (df_classification["avg_price"] - df_classification["price"])
    / df_classification["avg_price"]
)

df_classification["is_worth"] = (
    df_classification["value_percent"] >= 0.05
).astype(int)

feature_cols = [
    "gpu_tier",
    "cpu_tier",
    "ram_gb",
    "cpu_cores",
    "cpu_threads",
    "cpu_base_ghz",
    "cpu_boost_ghz",
    "price",
]


def load_classification_data():
    X = df_classification[feature_cols].values
    y = df_classification["is_worth"].values

    return X, y, feature_cols
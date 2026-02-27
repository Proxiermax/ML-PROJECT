from .raw_data import df
import numpy as np
import pandas as pd

df_regression = df.copy()

gap = 25
labels = np.arange(300, 11001, gap)

first_edge = labels[0] - gap / 2.0
last_edge = labels[-1] + gap / 2.0
bins_edges = np.arange(first_edge, last_edge + gap, gap)

df_regression['predicted_price'] = pd.cut(
    df_regression['price'],
    bins=bins_edges,
    labels=labels,
    right=True,
    include_lowest=True
).astype(float)


def encode_categorical_columns(df):
    df_encoded = df.copy()

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            df_encoded[col] = df_encoded[col].astype("category").cat.codes

    return df_encoded


def load_regression_data():

    df_encoded = encode_categorical_columns(df_regression)

    features = [
        'gpu_tier',
        'ram_gb',
        'resolution',
        'cpu_tier',
        'os',
        'cpu_threads',
        'cpu_cores'
    ]

    X = df_encoded[features]
    y = df_encoded['predicted_price']

    return X.values, y.values
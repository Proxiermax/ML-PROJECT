from .raw_data import df
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df_regression = df.copy()

gap = 25
labels = np.arange(300, 11001, gap)

first_edge = labels[0] - gap / 2.0
last_edge = labels[-1] + gap / 2.0
bins_edges = np.arange(first_edge, last_edge + gap, gap)

df_regression['predicted_price'] = pd.cut(df_regression['price'], bins=bins_edges, labels=labels, right=True, include_lowest=True).astype(float)

def load_regression_data():

    df_encoded = df_regression.copy()

    features = ['gpu_tier', 'ram_gb', 'resolution', 'cpu_tier', 'os', 'cpu_threads', 'cpu_cores']

    categorical_cols = ['gpu_tier', 'resolution', 'os']

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    df_encoded[categorical_cols] = encoder.fit_transform(df_encoded[categorical_cols])

    X = df_encoded[features]
    y = df_encoded['predicted_price']

    return X.values, y.values, encoder
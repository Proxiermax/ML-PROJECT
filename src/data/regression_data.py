from .raw_data import df
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df_regression = df.copy()

# ----------- สร้าง predicted_price -----------
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


# ----------- Encode non-float columns -----------
def encode_categorical_columns(df):

    df_encoded = df.copy()

    for col in df_encoded.columns:

        if df_encoded[col].dtype == 'object' or str(df_encoded[col].dtype).startswith('category'):
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    return df_encoded


def load_regression_data():

    df_encoded = encode_categorical_columns(df_regression)

    X = df_encoded.drop(columns=['predicted_price', 'price'])
    y = df_encoded['predicted_price']

    return X.values, y.values, X.columns.tolist()
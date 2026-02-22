from raw_data import df
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
)

print(df_regression.head())
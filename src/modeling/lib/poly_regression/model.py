from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


def create_poly_regression(degree=2):
    """Create a scikit-learn Pipeline with PolynomialFeatures + LinearRegression."""
    return Pipeline([
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=True)),
        ("linear_reg", LinearRegression()),
    ])

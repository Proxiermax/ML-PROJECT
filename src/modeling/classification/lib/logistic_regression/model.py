from sklearn.linear_model import LogisticRegression


def create_logistic_regression(max_iter=1000, random_state=42):
    return LogisticRegression(max_iter=max_iter, random_state=random_state)

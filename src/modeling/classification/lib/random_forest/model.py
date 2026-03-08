from sklearn.ensemble import RandomForestClassifier


def create_random_forest(n_estimators=50, max_depth=10, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )

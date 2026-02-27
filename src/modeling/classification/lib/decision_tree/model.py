from sklearn.tree import DecisionTreeClassifier


def create_decision_tree(max_depth=10, min_samples_split=5, criterion="gini",
                         random_state=42):
    """Create a scikit-learn DecisionTreeClassifier."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        criterion=criterion,
        random_state=random_state,
    )

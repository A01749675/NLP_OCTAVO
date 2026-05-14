from sklearn.ensemble import RandomForestClassifier


def get_model(random_state=42):
    """
    Returns a Random Forest Classifier configured for the project.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility. Defaults to 42.

    Returns
    -------
    sklearn.ensemble.RandomForestClassifier
        Configured Random Forest classifier instance.
    """
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced"
    )
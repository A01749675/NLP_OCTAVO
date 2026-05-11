from sklearn.ensemble import RandomForestClassifier


def get_model(random_state=42):
    """
        Returns a Random Forest Classifier with 100 trees, balanced class weights, and a fixed random state for reproducibility.

    Args:
        random_state (int, optional): _description_. Defaults to 42.

    Returns:
        sklearn model: A Random Forest Classifier instance.
    """
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced"
    )
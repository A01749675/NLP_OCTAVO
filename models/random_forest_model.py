from sklearn.ensemble import RandomForestClassifier


def get_model(random_state=42):
    return RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        class_weight="balanced"
    )
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def get_model(random_state=42):
    """
    Returns a Logistic Regression model pipeline with standard scaling.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducible results. Defaults to 42.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline containing a scaler and LogisticRegression classifier.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])
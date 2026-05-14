from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def get_model(n_neighbors=15):
    """
    Returns a KNN model pipeline with standard scaling.

    Parameters
    ----------
    n_neighbors : int, optional
        Number of neighbors to use in the KNN classifier. Defaults to 15.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A pipeline containing a scaler and KNeighborsClassifier.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="cosine"
        ))
    ])
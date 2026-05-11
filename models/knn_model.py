from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def get_model(n_neighbors=15):
    '''Returns a KNN model pipeline with standard scaling.
        The value of n_neighbors wwas determined through experimentation, with values between 1 and 21 showing good performance. 
        The "distance" weighting scheme was chosen to give more importance to closer neighbors, and the "cosine" metric 
        was selected based on its effectiveness in high-dimensional spaces, which is common in text data.
    '''
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="cosine"
        ))
    ])
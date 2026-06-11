"""Utilities for loading and preparing vectorized datasets.

This module reads CSV files produced by the vectorization pipeline,
validates the expected label column, and returns a feature matrix together
with the corresponding target labels.
"""

import pandas as pd

from paths import resolve_input_path
from scipy import sparse

import os


def get_data(vectorized_file):
    """Load a vectorized CSV file and split it into features and labels.

    Parameters
    ----------
    vectorized_file : str | pathlib.Path
        Path to a CSV file containing the vectorized features and a ``class``
        column. Relative paths are resolved through :func:`paths.resolve_input_path`.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        A tuple containing the numeric feature matrix ``X`` and the label series
        ``y``.

    Raises
    ------
    ValueError
        If the input file does not contain a ``class`` column or if no numeric
        feature columns are available after cleaning.
    """
    path = resolve_input_path(vectorized_file)

    # Support hashed .npz sparse outputs with a .meta.csv sidecar
    if str(path).lower().endswith('.npz'):
        X = sparse.load_npz(path)

        # load metadata sidecar
        meta_path = str(path).replace('.npz', '.meta.csv')
        if not os.path.exists(meta_path):
            raise ValueError(f"Metadata sidecar not found for {path}: expected {meta_path}")

        meta_df = pd.read_csv(meta_path, encoding='utf-8')

        if 'class' not in meta_df.columns:
            raise ValueError("The metadata file must contain a 'class' column.")

        y = meta_df['class']

        # Build sparse DataFrame with generated feature names so callers can access .columns
        n_features = X.shape[1]
        columns = [f"feat_{i}" for i in range(n_features)]
        X_df = pd.DataFrame.sparse.from_spmatrix(X, columns=columns)

        return X_df, y

    # Fallback: CSV path
    df = pd.read_csv(path, encoding="utf-8")

    if "class" not in df.columns:
        raise ValueError("The vectorized file must contain a 'class' column.")

    columns_to_drop = [
        "class",
        "tweet_id",
        "tweet_text",
        "tweet_text_clean",
        "user_id"
    ]

    X = df.drop(columns=columns_to_drop, errors="ignore")
    y = df["class"]

    X = X.select_dtypes(include=["number"]).copy()

    if X.empty:
        raise ValueError("No numeric feature columns were found.")

    return X, y
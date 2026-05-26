"""Utilities for loading and preparing vectorized datasets.

This module reads CSV files produced by the vectorization pipeline,
validates the expected label column, and returns a feature matrix together
with the corresponding target labels.
"""

import pandas as pd

from paths import resolve_input_path


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
    df = pd.read_csv(resolve_input_path(vectorized_file), encoding="utf-8")

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

    X = X.select_dtypes(include=["number"])

    if X.empty:
        raise ValueError("No numeric feature columns were found.")

    return X, y
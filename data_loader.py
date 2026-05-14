import pandas as pd


def get_data(vectorized_file):
    """
    Loads a vectorized dataset and separates features from labels.

    Parameters
    ----------
    vectorized_file : str
        Path to a CSV file that contains the vectorized features and a "class" column.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.Series]
        Feature matrix X and label vector y.
    """
    df = pd.read_csv(vectorized_file, encoding="utf-8")

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
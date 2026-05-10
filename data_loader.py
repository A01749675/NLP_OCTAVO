import pandas as pd


def get_data(vectorized_file):
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
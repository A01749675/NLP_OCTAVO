import ollama
import pandas as pd
from paths import resolve_input_path, resolve_output_path


def read_tweets(file_path):
    """
    Reads the CSV and keeps tweet_id, tweet_text, and class.
    The original class is renamed to real_class.
    """
    file_path = resolve_input_path(file_path)

    df = pd.read_csv(file_path, encoding="utf-8")

    required_columns = ["tweet_id", "tweet_text", "class"]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {missing_columns}")

    df = df[required_columns].dropna()

    df = df.rename(columns={
        "class": "real_class"
    })

    return df


def classify_tweet(tweet_text):
    """
    Sends one tweet to Ollama and returns the predicted class.
    """
    response = ollama.chat(
        model="gemma2:9b",
        messages=[
            {
                "role": "user",
                "content": f"""
Classify the following tweet into exactly one of these classes:

control
anorexia

Return only the class name. Do not explain.

Tweet:
{tweet_text}
""",
            }
        ],
    )

    prediction = response["message"]["content"].strip().lower()

    if "anorexia" in prediction:
        return "anorexia"
    elif "control" in prediction:
        return "control"
    else:
        return "unknown"


def classify_tweets(df):
    """
    Classifies every tweet and adds the Ollama prediction
    as a new column called new_class.
    """
    new_classes = []

    for _, row in df.iterrows():
        tweet_id = row["tweet_id"]
        tweet_text = row["tweet_text"]
        real_class = row["real_class"]

        print(f"Classifying tweet_id: {tweet_id} | real class: {real_class}")
        
        print(f"Tweet text: {tweet_text}")

        new_class = classify_tweet(tweet_text)
        new_classes.append(new_class)
        print(f"Predicted class: {new_class}")

    result_df = df.copy()
    result_df["new_class"] = new_classes

    return result_df


def save_results(df, output_file):
    """
    Saves the results to a CSV file.
    Output columns:
    tweet_id, tweet_text, real_class, new_class
    """
    output_path = resolve_output_path(output_file)

    output_columns = ["tweet_id", "tweet_text", "real_class", "new_class"]

    df[output_columns].to_csv(output_path, index=False, encoding="utf-8")

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    input_file = "data_train(in).csv"
    output_file = "ollama_predictions.csv"

    tweets_df = read_tweets(input_file)

    results_df = classify_tweets(tweets_df)

    print(results_df.head())

    save_results(results_df, output_file)
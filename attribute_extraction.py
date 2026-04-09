import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorize(texts, output_file="tfidf_output.csv"):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()
    # Create DataFrame from TF-IDF matrix
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
    # Add the original tweet as the first column
    tfidf_df.insert(0, "tweet_text_clean", texts)
    # Save to CSV
    tfidf_df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"TF-IDF data saved to {output_file}")
    return tfidf_df

def process_csv(input_file):
    df = pd.read_csv(input_file, encoding="utf-8")
    tweets = df["tweet_text_clean"].tolist()
    tfidf_vectorize(tweets, "data_train_tfidf.csv")

print("Processing CSV for TF-IDF vectorization...")
process_csv("data_train_cleaned.csv")
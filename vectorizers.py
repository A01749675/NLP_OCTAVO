import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from text_cleaner import text_filtering


# ---------------------------------------------------------
# TF-IDF VECTORIZER
# ---------------------------------------------------------
def tfidf_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_tfidf.csv",
    ngram_range=(1, 1)
):
    """
    Generates a TF-IDF representation from cleaned tweet texts.

    Parameters
    ----------
    texts : list
        List of cleaned tweet texts.

    tweet_ids : list
        List of tweet IDs.

    classes : list, optional
        List of class labels.

    output_file : str
        Name of the output CSV file.

    ngram_range : tuple
        Range of n-grams to use in TF-IDF.

    Returns
    -------
    pd.DataFrame
        DataFrame containing TF-IDF features and metadata.
    """

    tfidf = TfidfVectorizer(ngram_range=ngram_range)

    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{name}" for name in feature_names]
    )

    tfidf_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        tfidf_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        tfidf_df.insert(0, "class", classes)

    tfidf_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"TF-IDF data saved to {output_file}")
    print(f"Number of TF-IDF features: {len(feature_names)}")

    return tfidf_df


# ---------------------------------------------------------
# N-GRAM VECTORIZER
# ---------------------------------------------------------
def ngram_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_ngrams.csv",
    ngram_range=(1, 3)
):
    """
    Generates a count-based n-gram representation from cleaned tweets.

    Parameters
    ----------
    texts : list
        List of cleaned tweet texts.

    tweet_ids : list
        List of tweet IDs.

    classes : list, optional
        List of class labels.

    output_file : str
        Name of the output CSV file.

    ngram_range : tuple
        Range of n-grams to use.

    Returns
    -------
    pd.DataFrame
        DataFrame containing n-gram features and metadata.
    """

    vectorizer = CountVectorizer(ngram_range=ngram_range)

    ngram_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    ngram_df = pd.DataFrame(
        ngram_matrix.toarray(),
        columns=[f"ngram_{name}" for name in feature_names]
    )

    ngram_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        ngram_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        ngram_df.insert(0, "class", classes)

    ngram_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"N-gram data saved to {output_file}")
    print(f"Number of n-gram features: {len(feature_names)}")

    return ngram_df


# ---------------------------------------------------------
# WORD2VEC VECTORIZER
# ---------------------------------------------------------
def word2vec_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_word2vec.csv",
    vector_size=50,
    window=10,
    min_count=1,
    epochs=100
):
    """
    Generates one Word2Vec vector per tweet.

    This function trains a local Skip-gram Word2Vec model using:
    1. The cleaned tweets from the dataset.
    2. An additional domain-specific anorexia-related document.

    Each tweet is represented by averaging the vectors of its tokens.

    Parameters
    ----------
    texts : list
        List of cleaned tweet texts.

    tweet_ids : list
        List of tweet IDs.

    classes : list, optional
        List of class labels.

    output_file : str
        Name of the output CSV file.

    vector_size : int
        Number of dimensions for the Word2Vec embeddings.

    window : int
        Context window size.

    min_count : int
        Minimum frequency required for a word to be included.

    epochs : int
        Number of training epochs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Word2Vec features and metadata.
    """
    
    WORD_2_VEC_FILE = "word2vecText.txt"
    
    with open(WORD_2_VEC_FILE, "r", encoding="utf-8") as file:
            domain_document = file.read()
    # domain_document = text_filtering(domain_document)
    def clean_and_tokenize(text):
        text = str(text).lower()

        text = "".join(
            character for character in text
            if character.isalnum() or character.isspace()
        )

        tokens = word_tokenize(text)

        return tokens

    tweet_tokens = [clean_and_tokenize(text) for text in texts]

    domain_tokens = clean_and_tokenize(domain_document)

    training_corpus = tweet_tokens + [domain_tokens]

    model = Word2Vec(
        sentences=training_corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,
        workers=4,
        epochs=epochs
    )

    def tweet_to_vector(tokens):
        vectors = []

        for token in tokens:
            if token in model.wv:
                vectors.append(model.wv[token])

        if len(vectors) == 0:
            return np.zeros(vector_size)

        return np.mean(vectors, axis=0)

    tweet_vectors = [
        tweet_to_vector(tokens)
        for tokens in tweet_tokens
    ]

    feature_names = [
        f"word2vec_{index}"
        for index in range(vector_size)
    ]

    word2vec_df = pd.DataFrame(
        tweet_vectors,
        columns=feature_names
    )

    word2vec_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        word2vec_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        word2vec_df.insert(0, "class", classes)

    word2vec_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Word2Vec data saved to {output_file}")
    print(f"Number of Word2Vec dimensions: {vector_size}")

    return word2vec_df


def tfidf_ngram_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_tfidf_ngrams.csv",
    tfidf_ngram_range=(1, 1),
    count_ngram_range=(3, 3)
):
    """
    Combines TF-IDF and n-grams into one representation.
    """

    tfidf_df = tfidf_vectorize(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file="temporary_tfidf.csv",
        ngram_range=tfidf_ngram_range
    )

    ngram_df = ngram_vectorize(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file="temporary_ngrams.csv",
        ngram_range=count_ngram_range
    )

    combined_df = pd.concat(
        [
            tfidf_df.drop(columns=["tweet_text_clean", "tweet_id", "class"], errors="ignore"),
            ngram_df.drop(columns=["tweet_text_clean", "tweet_id", "class"], errors="ignore")
        ],
        axis=1
    )

    combined_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        combined_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        combined_df.insert(0, "class", classes)

    combined_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Combined TF-IDF + N-gram data saved to {output_file}")
    print(f"Number of TF-IDF features: {tfidf_df.shape[1] - 3}")
    print(f"Number of N-gram features: {ngram_df.shape[1] - 3}")
    print(f"Total features: {combined_df.shape[1] - 3}")

    return combined_df


def all_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_all.csv",
    tfidf_ngram_range=(1, 1),
    count_ngram_range=(3, 3)
):
    """
    Combines TF-IDF, n-grams, and Word2Vec into one representation.
    """

    # -----------------------------
    # 1. TF-IDF
    # -----------------------------
    tfidf = TfidfVectorizer(ngram_range=tfidf_ngram_range)
    tfidf_matrix = tfidf.fit_transform(texts)
    tfidf_features = tfidf.get_feature_names_out()

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{name}" for name in tfidf_features]
    )

    # -----------------------------
    # 2. N-grams
    # -----------------------------
    count = CountVectorizer(ngram_range=count_ngram_range)
    count_matrix = count.fit_transform(texts)
    count_features = count.get_feature_names_out()

    ngram_df = pd.DataFrame(
        count_matrix.toarray(),
        columns=[f"ngram_{name}" for name in count_features]
    )

    # -----------------------------
    # 3. Word2Vec
    # -----------------------------
    word2vec_df = word2vec_vectorize(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file="temporary_word2vec.csv"
    )

    columns_to_drop = [
        "class",
        "tweet_id",
        "tweet_text_clean"
    ]

    word2vec_features_df = word2vec_df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # -----------------------------
    # 4. Combine
    # -----------------------------
    all_df = pd.concat(
        [
            tfidf_df.reset_index(drop=True),
            ngram_df.reset_index(drop=True),
            word2vec_features_df.reset_index(drop=True)
        ],
        axis=1
    )

    all_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        all_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        all_df.insert(0, "class", classes)

    all_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Combined TF-IDF + N-gram + Word2Vec data saved to {output_file}")
    print(f"Number of TF-IDF features: {len(tfidf_features)}")
    print(f"Number of N-gram features: {len(count_features)}")
    print(f"Number of Word2Vec features: {word2vec_features_df.shape[1]}")
    print(f"Total features: {all_df.shape[1] - 3}")

    return all_df
# ---------------------------------------------------------
# PROCESS CSV
# ---------------------------------------------------------
def process_csv(input_file, target):
    """
    Reads a cleaned CSV file and generates the selected vectorized file.

    Available targets:
    - "tfidf"
    - "ngrams"
    - "word2vec"
    - "all"

    Parameters
    ----------
    input_file : str
        CSV file containing tweet_text_clean and class.

    target : str
        Type of vectorization to apply.

    Returns
    -------
    str
        Name of the generated vectorized CSV file.
    """

    df = pd.read_csv(input_file, encoding="utf-8")

    if "tweet_text_clean" not in df.columns:
        raise ValueError("The input file must contain a 'tweet_text_clean' column.")

    if "class" not in df.columns:
        raise ValueError("The input file must contain a 'class' column.")

    texts = df["tweet_text_clean"].fillna("").tolist()
    tweet_ids = df["tweet_id"].tolist() if "tweet_id" in df.columns else None
    classes = df["class"].tolist()

    match target:
        case "tfidf":
            file_name = "data_train_tfidf.csv"

            tfidf_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                ngram_range=(1, 1)
            )

        case "ngrams":
            file_name = "data_train_ngrams.csv"

            ngram_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                ngram_range=(3, 3)
            )

        case "word2vec":
            file_name = "data_train_word2vec.csv"

            word2vec_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                vector_size=50,
                window=10,
                min_count=1,
                epochs=100
            )

        case "all":
            file_name = "data_train_all.csv"

            all_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                tfidf_ngram_range=(1, 1),
                count_ngram_range=(3, 3)
            )
        case "tfidf_ngrams":
            file_name = "data_train_tfidf_ngrams.csv"

            tfidf_ngram_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                tfidf_ngram_range=(1, 1),
                count_ngram_range=(3, 3)
            )
        case _:
            raise ValueError(
                "Invalid target. Use 'tfidf', 'ngrams', 'word2vec', or 'all'."
            )

    return file_name


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    generated_file = process_csv(
        input_file="data_train_cleaned.csv",
        target="word2vec"
    )

    print(f"Generated file: {generated_file}")
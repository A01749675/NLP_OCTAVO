import pandas as pd
import re
import string

#stopwords
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import Stemmer
import spacy


# 1. Initialize the stemmer for a specific language
stemmer = Stemmer.Stemmer('spanish')
nlp = spacy.load('es_core_news_sm')
nltk.download('stopwords')
nltk.download('punkt')


def clean_text(text):
    if pd.isna(text):
        return text

    text = str(text).lower()                    # Lowercase
    text = re.sub(r"http(s*)\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"#\w*", "", text)            # Remove hashtags first
    text = re.sub(r"\d+", "", text)             # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r"\W+", " ", text)            # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces

    return text

def stopword_remover_nltk(text):
    stop_word_set = set(stopwords.words('spanish'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_word_set]
    return ' '.join(filtered_tokens)

def text_stemming(text):
    stemmer = SnowballStemmer('spanish')
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def text_lemmatization(text):
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)

def text_stemming_pystemmer(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stemWord(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def text_filtering(text):
    text = clean_text(text)
    text = stopword_remover_nltk(text)
    # text = text_stemming(text)
    text = text_lemmatization(text)
    return text

def process_csv(input_file, output_file, text_column="tweet_text"):
    df = pd.read_csv(input_file, encoding="utf-8")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {list(df.columns)}")

    df[f"{text_column}_clean"] = df[text_column].apply(text_filtering)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Saved cleaned CSV to: {output_file}")

if __name__ == "__main__":
    process_csv("data_train(in).csv", "data_train_cleaned.csv", text_column="tweet_text")
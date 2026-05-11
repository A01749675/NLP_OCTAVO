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
    """
    Cleans the input text by performing several preprocessing steps:
    - Converts text to lowercase
    - Removes URLs, hashtags, numbers, mentions, punctuation, special characters, and extra spaces.
    - Handles missing values by returning them unchanged.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: the cleaned text.
    """
    if pd.isna(text):
        return text

    text = str(text).lower()                    # Lowercase
    text = re.sub(r"http(s*)\S+|www\S+|https\S+", "", text)  # Remove URLs
    text = re.sub(r"#\w*", "", text)            # Remove hashtags first
    text = re.sub(r"\d+", "", text)             # Remove numbers
    text = re.sub(r"@\w*", "", text)            # Remove mentions
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r"\W+", " ", text)            # Remove special chars
    text = re.sub(r"\s+", " ", text).strip()    # Remove extra spaces

    return text

def stopword_remover_nltk(text):
    """
    Removes the stopwords in spanish for the given text

    Args:
        text (str): input text to remove stopwords from

    Returns:
        str: text with stopwords removed
    """
    stop_word_set = set(stopwords.words('spanish'))
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word not in stop_word_set]
    return ' '.join(filtered_tokens)

def text_stemming(text):
    """ 
        Stems the input text using the Snowball Stemmer for Spanish.
        Args:
            text (str): The input text to be stemmed.
        Returns:
            str: The stemmed version of the input text.
    """
    stemmer = SnowballStemmer('spanish')
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def text_lemmatization(text):
    """ 
        Lemmatizes the input text using spaCy's Spanish model.
        Args:
            text (str): The input text to be lemmatized.
        Returns:
            str: The lemmatized version of the input text.
    """
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return ' '.join(lemmatized_tokens)

def text_stemming_pystemmer(text):
    """ 
        Stems the input text using the PyStemmer library for Spanish.
        Args:
            text (str): The input text to be stemmed.
        Returns:
            str: The stemmed version of the input text.
    
    """
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stemWord(word) for word in tokens]
    return ' '.join(stemmed_tokens)

def text_filtering(text):
    """ 
        Applies a series of text preprocessing steps to clean and normalize the input text.
        The steps include cleaning the text, removing stopwords, and applying stemming.
        Args:
            text (str): The input text to be processed.
        Returns:
            str: The processed text after cleaning, stopword removal, and stemming.
            
    """
    text = clean_text(text)
    text = stopword_remover_nltk(text)
    text = text_stemming(text)
    #text = text_lemmatization(text)
    return text

def process_csv(input_file, output_file, text_column="tweet_text"):
    """
    Receives an input csv file with the raw data of the text and generates a new file with the cleaned text field

    Args:
        input_file (str): name of the input csv file containing the raw text data.
        output_file (str): name of the output csv file where the cleaned text data will be saved.
        text_column (str, optional): _description_. Defaults to "tweet_text".

    Raises:
        ValueError: _description_
    """
    df = pd.read_csv(input_file, encoding="utf-8")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found. Available columns: {list(df.columns)}")

    df[f"{text_column}_clean"] = df[text_column].apply(text_filtering)

    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Saved cleaned CSV to: {output_file}")

if __name__ == "__main__":
    process_csv("data_train(in).csv", "data_train_cleaned.csv", text_column="tweet_text")
# NLP_OCTAVO

## Project Overview

`NLP_OCTAVO` is a Python project for binary text classification using Spanish tweets. It provides a pipeline for cleaning raw tweet data, generating multiple text feature representations, and training machine learning classifiers to detect the target class.

The repository supports:
- text cleaning and preprocessing in Spanish
- TF-IDF feature extraction
- count-based n-gram extraction
- Word2Vec embedding creation using a domain-specific corpus
- classification using Random Forest, Logistic Regression, and K-Nearest Neighbors
- model evaluation with accuracy, precision, recall, F1, specificity, and AUC metrics

## Key Components

### Data preprocessing
- `text_cleaner.py`
  - cleans text by removing URLs, hashtags, mentions, punctuation, numbers, and extra whitespace
  - removes Spanish stopwords using NLTK
  - applies stemming for Spanish text
  - supports optional lemmatization via spaCy
- `data_loader.py`
  - loads vectorized CSV files and returns numeric feature matrices plus class labels

### Vectorization
- `vectorizers.py`
  - `tfidf_vectorize()` generates TF-IDF features
  - `ngram_vectorize()` generates count-based n-gram features
  - `word2vec_vectorize()` generates averaged Word2Vec vectors per tweet using an additional domain-specific document

### Models
- `models/random_forest_model.py`
  - Random Forest classifier with `class_weight='balanced'`
- `models/logistic_regression_model.py`
  - Logistic Regression pipeline with scaling and balanced class weights
- `models/knn_model.py`
  - KNN pipeline with scaling, cosine distance, and distance weighting

### Evaluation
- `evaluation.py`
  - computes accuracy, precision, recall, F1 macro, specificity
  - generates confusion matrix plots, ROC curves, and feature importance plots where applicable

### Entry points
- `main.py`
  - orchestrates the full training pipeline
  - supports `train_and_plot()` for TF-IDF / n-gram / combined feature training
  - includes `test_knn_model()` for testing KNN across multiple vector representations and neighbor counts

## Data Files

- `data_train(in).csv` - raw input dataset
- `data_train_cleaned.csv` - cleaned dataset output by preprocessing
- `data_train_tfidf.csv` - TF-IDF features
- `data_train_ngrams.csv` - n-gram count features
- `data_train_word2vec.csv` - Word2Vec features
- `data_train_all.csv` - combined representation dataset
- `word2vecText.txt` - domain-specific text used to train Word2Vec embeddings

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> Note: `text_cleaner.py` uses NLTK and spaCy Spanish components. Ensure `nltk` downloads and the `es_core_news_sm` spaCy model are available.

## Usage

1. Clean the raw input data:

```bash
python text_cleaner.py
```

2. Vectorize the cleaned dataset manually or via `main.py`.

3. Run training and evaluation from `main.py` by importing its functions or adding your own driver logic.

Example in Python:

```python
from main import train_and_plot

train_and_plot(
    input_file="data_train_cleaned.csv",
    target="tfidf",
    model_name="rf",
    random_state=42
)
```

Example for KNN experiments:

```python
from main import test_knn_model

test_knn_model(input_file="data_train_cleaned.csv")
```

## Project Structure

- `main.py` - experiment orchestration and training logic
- `data_loader.py` - dataset loading helper
- `text_cleaner.py` - Spanish text cleaning pipeline
- `vectorizers.py` - TF-IDF, n-gram, and Word2Vec vectorization
- `evaluation.py` - metrics and visualization utilities
- `models/` - model definitions for Random Forest, Logistic Regression, and KNN
- `legacy/` - legacy or experimental code
- `tests/` - placeholder for unit / integration tests

## Notes

- The repository is specially focused on classification experiments for Spanish-language tweets.
- Current model evaluation supports binary classification and assumes a `class` column in vectorized CSV files.
- The dataset contains metadata columns such as `tweet_id`, `tweet_text`, `tweet_text_clean`, and `user_id`.

## License

This repository does not include a license file. Add one if you plan to share or publish the code.

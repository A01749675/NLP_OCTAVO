# NLP_OCTAVO

## Project Overview

`NLP_OCTAVO` is a Python project for Spanish tweet classification. It currently includes:

- text cleaning and preprocessing
- multiple feature extraction strategies
- model training for Random Forest, Logistic Regression, and KNN
- evaluation utilities
- a validation script that evaluates saved model artifacts against a cleaned test set

## Current pipeline

### Data preprocessing
- `text_cleaner.py` performs the current cleaning flow used by `main.py` and `model_validation.py`.
- The preprocessing pipeline removes URLs, hashtags, mentions, numbers, punctuation, extra whitespace, Spanish stopwords, and applies stemming.
- The `text_lemmatization()` helper exists, but it is not part of the active `text_filtering()` pipeline.

### CSV path handling
- `paths.py` centralizes input and output path resolution.
- Generated CSVs are written under the `files/` directory.
- `resolve_input_path()` checks `files/` first and then falls back to the repository root for backward compatibility.
- `resolve_output_path()` always writes under `files/` and creates the folder if needed.

### Vectorization
- `vectorizers.py` exposes the current vectorizers and routing helpers:
  - `tfidf_vectorize()`
  - `ngram_vectorize()`
  - `word2vec_vectorize()`
  - `tfidf_bigrams_vectorize()`
  - `tfidf_trigrams_vectorize()`
  - `all_vectorize()`
  - `process_csv()`

### Supported `process_csv()` targets
- `tfidf`
- `ngrams`
- `bigrams`
- `trigrams`
- `word2vec`
- `all`
- `tfidf_bigrams`
- `tfidf_trigrams`

The combined TF-IDF variants currently use these settings:
- `tfidf_bigrams`: TF-IDF range `(1, 2)` + count range `(2, 2)`
- `tfidf_trigrams`: TF-IDF range `(1, 3)` + count range `(3, 3)`

### Models
- `models/random_forest_model.py` -> Random Forest classifier with balanced class weights
- `models/logistic_regression_model.py` -> Logistic Regression pipeline with scaling
- `models/knn_model.py` -> KNN pipeline with scaling, cosine distance, and distance weighting

### Evaluation
- `evaluation.py` computes accuracy, precision, recall, F1, specificity, and ROC AUC where supported.
- It also contains plotting helpers for class distribution, confusion matrices, feature importance, and ROC curves.

### Validation
- `model_validation.py` cleans the test file, vectorizes it using the saved model target, aligns the test features to the training columns, evaluates the saved artifacts, and writes `model_evaluation_results.csv`.

## Key files

- `main.py` -> orchestration for training experiments and saving model artifacts
- `vectorizers.py` -> vectorization implementations and target routing
- `data_loader.py` -> loads vectorized CSV files and returns `(X, y)`
- `text_cleaner.py` -> text preprocessing helpers
- `evaluation.py` -> metrics and plotting helpers
- `model_validation.py` -> validation of saved `.pkl` model artifacts
- `tests/` -> unit tests for the vectorization and routing logic

## Data files

### Source inputs
- `files/data_train(in).csv` -> raw input dataset used by the current pipeline
- `files/data_test_fold1(in).csv` -> raw test dataset used by validation
- `files/data_train_cleaned.csv` -> cleaned dataset output used by the training pipeline

### Generated artifacts
- `files/data_train_tfidf.csv` -> TF-IDF features
- `files/data_train_ngrams.csv` -> generic n-gram features
- `files/data_train_bigrams.csv` -> bigram features
- `files/data_train_trigrams.csv` -> trigram features
- `files/data_train_word2vec.csv` -> Word2Vec features
- `files/data_train_all.csv` -> combined TF-IDF + n-gram + Word2Vec representation
- `files/data_train_tfidf_bigrams.csv` -> combined TF-IDF + bigram representation
- `files/data_train_tfidf_trigrams.csv` -> combined TF-IDF + trigram representation
- `files/model_evaluation_results.csv` -> generated output from `model_validation.py`
- `files/temporary_tfidf.csv` and `files/temporary_ngrams.csv` -> temporary vectorization outputs used during validation and tests

### Model artifacts
- `WORD2VEC.model` -> cached Word2Vec model used by `word2vec_vectorize()`
- `word2vecText.txt` -> auxiliary domain text used for Word2Vec training

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

> `text_cleaner.py` downloads NLTK resources at import time and expects the Spanish spaCy model `es_core_news_sm` to be available.

## Usage

### Clean the raw data

```bash
python text_cleaner.py
```

This uses `paths.py` to read `files/data_train(in).csv` and write `files/data_train_cleaned.csv`.

### Run the full experiment suite

```bash
python main.py
```

`main.py` calls `run_experiments()` and saves model artifacts as `model-representation.pkl` files in the repository root.

### Vectorize a cleaned CSV manually

```python
from vectorizers import process_csv

process_csv("files/data_train_cleaned.csv", "tfidf")
process_csv("files/data_train_cleaned.csv", "tfidf_bigrams")
process_csv("files/data_train_cleaned.csv", "word2vec")
```

### Train a single model configuration

```python
from main import train_and_plot

train_and_plot(
    input_file="files/data_train_cleaned.csv",
    target="tfidf",
    model_name="rf",
    random_state=42
)
```

### Run KNN experiments

```python
from main import test_knn_model

test_knn_model(input_file="files/data_train_cleaned.csv")
```

### Validate saved models

```bash
python model_validation.py
```

This reads the current `MODELS` list from `model_validation.py`, cleans `files/data_test_fold1(in).csv`, vectorizes the test file using the saved model target, aligns any missing feature columns to zero, evaluates each saved artifact, and exports `files/model_evaluation_results.csv`.

## Current experiment coverage

`run_experiments()` currently includes:

- `tfidf`
- `bigrams`
- `trigrams`
- `word2vec`
- `all`
- `tfidf_bigrams`
- `tfidf_trigrams`

The model validation script evaluates the corresponding saved artifacts from `MODELS`.

## Notes

- The repository is focused on Spanish-language tweet classification.
- `process_csv()` requires a `tweet_text_clean` column and a `class` column in the input CSV.
- The current validation flow uses `feature_columns` stored in each saved model artifact to align the test frame before prediction.
- The `tfidf_ngram_vectorize()` helper is still present as a backward-compatible alias for the trigram-based combined output.

## License

This repository does not include a license file. Add one if you plan to share or publish the code.

# NLP_OCTAVO

## Project Overview

`NLP_OCTAVO` is a Spanish tweet classification toolkit built in Python. The repository now includes:

- text cleaning and preprocessing
- feature extraction for TF-IDF, n-grams, Word2Vec, BETO, and RoBERTuito
- training pipelines for Random Forest, Logistic Regression, and KNN
- model evaluation and validation utilities
- AI classification using Ollama-based LLM prompts
- dedicated fine-tuning scripts for BETO and RoBERTuito

## Current architecture and execution flow

The active flow consists of two main branches:

1. Traditional supervised learning pipeline (`main.py`)
2. AI classifier / prompt-based inference flow (`ai_classifier.py`)

The traditional supervised workflow starts with raw CSV input, which is cleaned by `text_cleaner.py`. The cleaned dataset is then vectorized by `vectorizers.py` through `process_csv()`, producing feature CSVs for TF-IDF, n-grams, Word2Vec, or transformer-ready representations. `main.py` loads these vectorized files, trains the selected model, saves the artifact, and uses `evaluation.py` for metrics and plotting. After training, `model_validation.py` can load the saved artifacts, align feature columns, and validate performance on cleaned test data.

The AI classifier branch is separate from the classical training flow. `ai_classifier.py` reads tweets, sends them to Ollama using strict prompt templates, and evaluates predictions against the same classification labels. This branch is intended for prompt-based inference rather than conventional model training.

Transformer fine-tuning is handled by `main_bert.py` and `main_robertuito.py`. These scripts prepare a cleaned transformer dataset, load the corresponding base model, fine-tune it for the binary classification task, and save the best model artifacts.

## Key files and current responsibilities

- `text_cleaner.py` — main preprocessing pipeline for training and validation data
- `paths.py` — centralized path resolution for inputs, outputs, and models
- `vectorizers.py` — converts cleaned data into TF-IDF, n-grams, Word2Vec, and transformer-ready representations
- `data_loader.py` — loads feature CSVs and returns `(X, y)` for training
- `main.py` — orchestrates training experiments and saves model artifacts
- `evaluation.py` — computes metrics, confusion matrices, ROC AUC, and plotting helpers
- `model_validation.py` — validates saved `.pkl` models against a cleaned test set
- `ai_classifier.py` — prompt-based AI classification using Ollama and specialized prompt styles
- `main_bert.py` — fine-tunes BETO for the tweet classification task
- `main_robertuito.py` — fine-tunes RoBERTuito for the tweet classification task

## Supported training representations

The supervised training pipeline currently supports:

- `tfidf`
- `ngrams`
- `bigrams`
- `trigrams`
- `word2vec`
- `all`
- `tfidf_bigrams`
- `tfidf_trigrams`
- `beto`
- `beto_finetuned`
- `robertuito`
- `robertuito_finetuned`

## AI classifier implementation

`ai_classifier.py` adds an alternative inference path using Ollama.

It exposes:

- `read_tweets(file_path)` — loads tweet records and required columns
- `classify_tweet(tweet_text, model)` — single-shot prompt classification
- `classify_tweet_few_shot(tweet_text, model)` — few-shot prompt classification
- `classify_chain_of_thought(tweet_text, model)` — chain-of-thought prompt variation
- `calculate_confusion_matrix(output_file)` — evaluation for prompt-based predictions

The AI classifier routes tweets through an LLM prompt and enforces a strict output of either `control` or `anorexia`.

## Fine-tuning scripts

- `main_bert.py` — loads `dccuchile/bert-base-spanish-wwm-cased`, tokenizes cleaned data, fine-tunes for 2 classes, saves best model to `modelo_beto_final/`, and writes metrics to `metricas_evaluacion_entrenamiento.txt`
- `main_robertuito.py` — loads `pysentimiento/robertuito-sentiment-analysis`, adapts the head to 2 classes, fine-tunes, saves best model to `modelo_robertuito_final/`, and writes metrics to `metricas_evaluacion_robertuito.txt`

## Path and artifact conventions

- Input CSVs live under `files/` or at repository root for compatibility.
- Cleaned data is written to `files/data_train_cleaned.csv` and `files/data_train_cleaned2.csv`.
- Vectorized outputs are written to `files/` as produced by `process_csv()`.
- Saved traditional ML artifacts are written as `model_name-target.pkl` via `main.py`.
- Fine-tuned transformer models are saved under `modelo_beto_final/` and `modelo_robertuito_final/`.

## Usage examples

### Install requirements

```bash
pip install -r requirements.txt
```

### Clean the raw training data

```bash
python text_cleaner.py
```

### Run the full supervised experiment suite

```bash
python main.py
```

### Run a specific experiment from Python

```python
from main import train_and_plot

train_and_plot(
    input_file="files/data_train_cleaned.csv",
    target="tfidf",
    model_name="rf",
    random_state=42
)
```

### Run KNN sweep experiments

```python
from main import test_knn_model

test_knn_model(input_file="files/data_train_cleaned.csv")
```

### Validate saved model artifacts

```bash
python model_validation.py
```

### Run prompt-based AI classification

```python
from ai_classifier import read_tweets, classify_tweet, calculate_confusion_matrix

# load tweets and classify
``` 

> `ai_classifier.py` uses the Ollama client and requires Ollama to be installed and configured locally.

### Fine-tune transformer models

```bash
python main_bert.py
python main_robertuito.py
```

## Generated outputs

- `files/model_evaluation_results.csv` — validated saved model results
- `files/all_experiments.csv` — results from `run_experiments()`
- `files/beto_experiments.csv` — results from `run_beto()`
- `files/knn_performance.csv` — results from `test_knn_model()`
- `metricas_evaluacion_entrenamiento.txt` — BETO training metrics
- `metricas_evaluacion_robertuito.txt` — RoBERTuito training metrics

## Notes

- `text_cleaner.py` downloads required NLTK resources and expects Spanish NLP support.
- Transformer fine-tuning is GPU-friendly but falls back to CPU if CUDA is unavailable.
- `main.py` discriminates between cleaned sets: `files/data_train_cleaned.csv` for classical features and `files/data_train_cleaned2.csv` for transformer-based targets.
- The repository is centered on Spanish tweet classification for `control` vs `anorexia` labels.

## License

This repository does not include a license file. Add one before sharing publicly.

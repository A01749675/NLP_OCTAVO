"""Model validation pipeline for NLP classifiers.

This module loads trained model artifacts, cleans and vectorizes the test
dataset, aligns the feature columns with the ones used during training, and
computes evaluation metrics for each model.

The main entry point is ``run_model_validation()``, which writes a CSV report
to the configured output path.
"""
from text_cleaner import *
import os
import joblib
import pandas as pd

from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score

from paths import resolve_input_path, resolve_model_path, resolve_output_path
from vectorizers import process_csv
from data_loader import get_data


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
#
INPUT_TEST_FILE = os.path.join("files", "data_test_fold1(in).csv")
CLEANED_TEST_FILE = os.path.join("files", "cleaned_data_test_fold1(in).csv")
CLEANED_TEST_FILE_2 = os.path.join("files", "cleaned_data_test_fold1(in)2.csv")
OUTPUT_RESULTS_FILE = resolve_output_path("model_evaluation_results.csv")

TEXT_COLUMN = "tweet_text"
LABEL_COLUMN = "class"

MODELS = [
    "knn-tfidf.pkl",
    "lr-tfidf.pkl",
    "rf-tfidf.pkl",

    "knn-word2vec.pkl",
    "lr-word2vec.pkl",
    "rf-word2vec.pkl",

    "knn-trigrams.pkl",
    "lr-trigrams.pkl",
    "rf-trigrams.pkl",

    "knn-bigrams.pkl",
    "lr-bigrams.pkl",
    "rf-bigrams.pkl",

    "knn-all.pkl",
    "lr-all.pkl",
    "rf-all.pkl",

    "knn-tfidf_bigrams.pkl",
    "lr-tfidf_bigrams.pkl",
    "rf-tfidf_bigrams.pkl",

    "knn-tfidf_trigrams.pkl",
    "lr-tfidf_trigrams.pkl",
    "rf-tfidf_trigrams.pkl",
    
    "knn-beto.pkl",
    "lr-beto.pkl",
    "rf-beto.pkl"
]


# ---------------------------------------------------------
# Cleaning
# ---------------------------------------------------------

def clean_data(
    input_file=INPUT_TEST_FILE,
    output_file=CLEANED_TEST_FILE,
    text_column=TEXT_COLUMN
):
    """
    Reads the test CSV, applies text cleaning, and saves a cleaned CSV file.
    
    args:
        input_file (str): Path to the raw test CSV file.
        output_file (str): Path where the cleaned CSV will be saved.
        text_column (str): Name of the column containing the text to clean.
        
    returns:
        str: Path to the cleaned CSV file.
        
    """

    input_path = resolve_input_path(input_file)
    output_path = resolve_output_path(output_file)

    df = pd.read_csv(input_path, encoding="utf-8")

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' was not found. Available columns: {list(df.columns)}"
        )

    df["tweet_text_clean"] = df[text_column].fillna("").apply(text_filtering)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{LABEL_COLUMN}' was not found. Available columns: {list(df.columns)}"
        )

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Cleaned data saved to {output_path}")

    return output_path



def clean_data_2(
    input_file=INPUT_TEST_FILE,
    output_file=CLEANED_TEST_FILE_2,
    text_column=TEXT_COLUMN
):
    """
    Reads the test CSV, applies text cleaning, and saves a cleaned CSV file.
    
    args:
        input_file (str): Path to the raw test CSV file.
        output_file (str): Path where the cleaned CSV will be saved.
        text_column (str): Name of the column containing the text to clean.
        
    returns:
        str: Path to the cleaned CSV file.
        
    """

    input_path = resolve_input_path(input_file)
    output_path = resolve_output_path(output_file)

    df = pd.read_csv(input_path, encoding="utf-8")

    if text_column not in df.columns:
        raise ValueError(
            f"Column '{text_column}' was not found. Available columns: {list(df.columns)}"
        )

    df["tweet_text_clean"] = df[text_column].fillna("").apply(text_filtering2)

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{LABEL_COLUMN}' was not found. Available columns: {list(df.columns)}"
        )

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Cleaned data saved to {output_path}")

    return output_path


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def parse_model_filename(model_file):
    """
    Extracts model name and representation from filenames like:
    knn-tfidf.pkl
    lr-word2vec.pkl
    rf-tfidf_bigrams.pkl
    
    args:
        model_file (str): The filename of the model artifact.
    returns:
        tuple: (model_name, representation)
    """

    filename = os.path.basename(model_file)
    filename = filename.replace(".pkl", "")

    parts = filename.split("-", maxsplit=1)

    if len(parts) != 2:
        raise ValueError(
            f"Invalid model filename: {model_file}. Expected format: model-representation.pkl"
        )

    model_name = parts[0]
    representation = parts[1]

    return model_name, representation


def load_model_artifact(model_file):
    """
    Loads the saved artifact.

    New expected format:
    {
        "model": model,
        "model_name": model_name,
        "target": target,
        "feature_columns": [...]
    }

    Also supports older files that only contain the raw model,
    but those are less reliable.
    
    args:
        model_file (str): Path to the saved model artifact.
    returns:
        tuple: (model, model_name, target, feature_columns)
    """

    model_path = resolve_model_path(model_file)
    loaded = joblib.load(model_path)

    if isinstance(loaded, dict):
        model = loaded["model"]
        target = loaded.get("target")
        model_name = loaded.get("model_name")
        feature_columns = loaded.get("feature_columns")

        return model, model_name, target, feature_columns

    # Fallback for older .pkl files
    model_name, target = parse_model_filename(model_file)

    return loaded, model_name, target, None


def align_features_to_training(X_test, feature_columns):
    """
    Aligns validation/test features to the columns used during training.

    Missing columns are added as 0.
    Extra columns are removed.
    Column order is forced to match training.
    args:
        X_test (pd.DataFrame): The test features to align.
        feature_columns (list): The list of feature columns used during training.
    returns:
        pd.DataFrame: The aligned test features.
    """

    if feature_columns is None:
        print(
            "Warning: this model file does not contain feature_columns. "
            "Prediction may fail if test columns do not match training columns."
        )
        return X_test

    X_test = X_test.copy()

    X_test = X_test.reindex(columns=feature_columns, fill_value=0)

    return X_test


def get_positive_label(y_test, model):
    """
    Chooses the positive class for AUC and specificity.
    Prefers model.classes_[1] when available.
    
    args:
        y_test (array-like): The true labels for the test set.
        model: The trained model, which may have a classes_ attribute.
    returns:
        The label considered as the positive class.
    """

    if hasattr(model, "classes_") and len(model.classes_) == 2:
        return model.classes_[1]

    labels = sorted(pd.Series(y_test).dropna().unique())

    if len(labels) != 2:
        raise ValueError(f"Expected binary labels, got: {labels}")

    return labels[1]


def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model on aligned test features.
    
    args:
        model: The trained model to evaluate.
        X_test (pd.DataFrame): The test features, aligned to training columns.
        y_test (array-like): The true labels for the test set.
    returns:
        tuple: (y_pred, results_dict) where y_pred are the predicted labels and
               results_dict contains accuracy, precision_macro, recall_macro, f1_macro,
               specificity, and auc.
    """

    y_pred = model.predict(X_test)

    labels = sorted(pd.Series(y_test).dropna().unique())

    if len(labels) == 2:
        positive_label = get_positive_label(y_test, model)
        negative_label = [label for label in labels if label != positive_label][0]

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[negative_label, positive_label]
        )

        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = "N/A"

    auc = "N/A"

    if hasattr(model, "predict_proba") and len(labels) == 2:
        try:
            positive_label = get_positive_label(y_test, model)
            class_labels = list(model.classes_)
            positive_index = class_labels.index(positive_label)

            y_score = model.predict_proba(X_test)[:, positive_index]
            y_binary = (pd.Series(y_test) == positive_label).astype(int)

            auc = roc_auc_score(y_binary, y_score)

        except Exception as error:
            print(f"Could not calculate AUC: {error}")
            auc = "N/A"

    results = {
        "accuracy": metrics.accuracy_score(y_test, y_pred),
        "precision_macro": metrics.precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "recall_macro": metrics.recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "f1_macro": metrics.f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        ),
        "specificity": specificity,
        "auc": auc
    }

    return y_pred, results


# ---------------------------------------------------------
# Main validation
# ---------------------------------------------------------

def run_model_validation():
    """
    Cleans the test file, vectorizes it, aligns the features with training,
    evaluates every saved model, and exports one CSV with all metrics.
    
    returns:
        pd.DataFrame: A DataFrame containing the evaluation results for all models.
    """

    clean_data()
    clean_data_2()
    
    all_results = []

    for model_file in MODELS:
            
        
        model_file = resolve_model_path(model_file)
        if not os.path.exists(model_file):
            print(f"Skipping missing model file: {model_file}")
            continue

        model, artifact_model_name, artifact_target, feature_columns = load_model_artifact(model_file)

        file_model_name, file_target = parse_model_filename(model_file)

        model_name = artifact_model_name or file_model_name
        target = artifact_target or file_target

        print("\n" + "=" * 60)
        print(f"Evaluating model file: {model_file}")
        print(f"Model: {model_name}")
        print(f"Representation: {target}")
        print("=" * 60)
        if "beto" in model_file:
            vectorized_test_file = process_csv(
                input_file=CLEANED_TEST_FILE_2,
                target=target
            )
        else:
            vectorized_test_file = process_csv(
                input_file=CLEANED_TEST_FILE,
                target=target
            )

        X_test, y_test = get_data(vectorized_test_file)

        X_test = align_features_to_training(
            X_test=X_test,
            feature_columns=feature_columns
        )

        y_pred, results = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test
        )

        result_record = {
            "model_file": model_file,
            "representation": target,
            "model": model_name,
            **results
        }

        all_results.append(result_record)

        print(result_record)

    results_df = pd.DataFrame(all_results)

    results_df.to_csv(
        OUTPUT_RESULTS_FILE,
        index=False,
        encoding="utf-8"
    )

    print(f"\nSaved validation results to {OUTPUT_RESULTS_FILE}")

    return results_df


if __name__ == "__main__":
    run_model_validation()
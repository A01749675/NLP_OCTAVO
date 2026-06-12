"""Evaluation metrics for Gemma3 model predictions.

This module reads a CSV file containing true and predicted labels,
validates the required columns, and calculates classification metrics
including accuracy, precision, recall, f1-score, and specificity.
"""

import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


def load_gemma3_predictions(csv_path):
    """Load a CSV file with predictions and extract the labels.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the predictions.

    Returns
    -------
    tuple[pandas.Series, pandas.Series, pandas.Series or None]
        A tuple containing the true labels ``real_class``, the predicted 
        labels ``new_class`` as string series, and optionally probability scores 
        if available in the CSV.

    Raises
    ------
    ValueError
        If the input CSV does not contain the required 'real_class' and 
        'new_class' columns.
    """
    df = pd.read_csv(csv_path)

    # Verify required columns are present in the dataframe
    expected = {"real_class", "new_class"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"El archivo {csv_path} debe contener las columnas {expected}. "
            f"Columnas actuales: {list(df.columns)}"
        )
    
    # Try to load probability scores if available
    probs = None
    possible_prob_cols = ['probability', 'score', 'confidence', 'prob']
    for col in possible_prob_cols:
        if col in df.columns:
            probs = df[col]
            break
    
    return df["real_class"].astype(str), df["new_class"].astype(str), probs


def compute_specificity(y_true, y_pred, positive_label=None):
    """Calculate the specificity (true negative rate) for a binary classification.

    Parameters
    ----------
    y_true : array-like
        Ground truth (correct) target values.
    y_pred : array-like
        Estimated targets as returned by a classifier.
    positive_label : str, optional
        The class label to be treated as the positive class. If None, the first 
        label in the sorted unique classes is used.

    Returns
    -------
    float
        The calculated specificity. Returns 0.0 if the denominator is zero.

    Raises
    ------
    ValueError
        If the combination of true and predicted labels contains more or fewer 
        than exactly 2 unique classes.
    """
    labels = sorted(set(y_true) | set(y_pred))

    # Ensure the classification is strictly binary
    if len(labels) != 2:
        raise ValueError("La especificidad solo se calcula para clasificador binario.")

    if positive_label is None:
        positive_label = labels[0]

    # Identify the negative label dynamically based on the available classes
    negative_label = [label for label in labels if label != positive_label][0]
    cm = confusion_matrix(y_true, y_pred, labels=[positive_label, negative_label])
    tp, fn, fp, tn = cm.ravel()

    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate_gemma3(csv_path, model_name="gemma3", positive_label="anorexia"):
    """Evaluate predictions using accuracy, precision, recall, f1, specificity, and AUC.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing the predictions.
    model_name : str, optional
        The name of the model being evaluated (default is "gemma3").
    positive_label : str, optional
        The label used as the positive class for the specificity calculation
        (default is "anorexia").

    Returns
    -------
    dict
        A dictionary containing the model name and the computed evaluation metrics.
    """
    y_true, y_pred, y_proba = load_gemma3_predictions(csv_path)

    # Compute aggregate metrics using macro averaging for multiclass support (if applicable)
    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": compute_specificity(y_true, y_pred, positive_label=positive_label),
    }
    
    # Calculate AUC if probability scores are available
    if y_proba is not None:
        # Convert labels to binary (1 for positive_label, 0 otherwise)
        y_true_binary = (y_true == positive_label).astype(int)
        results["auc"] = roc_auc_score(y_true_binary, y_proba)
        print(f"AUC calculado para {model_name}: {results['auc']:.4f}")

    return results


def main():
    """Main entry point for the script to evaluate metrics via CLI."""
    parser = argparse.ArgumentParser(
        description="Calcula métricas de evaluación para un CSV de Gemma3."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        default=os.path.join("files", "gemma3_predictions.csv"),
        help="Ruta al CSV de Gemma3 (default: files/gemma3_predictions.csv)",
    )
    parser.add_argument(
        "--positive-label",
        default="anorexia",
        help="Etiqueta de la clase positiva utilizada para especificidad."
    )
    args = parser.parse_args()

    # Execute evaluation and convert the results into a DataFrame for standard display
    results = evaluate_gemma3(args.csv_path, model_name="llama3:2latest", positive_label=args.positive_label)
    df = pd.DataFrame([results])

    print(df.to_string(index=False))

def check_files():
    
    files_to_check = ['test_1_llama_32_predictions_few_fold1_False.csv', 
                      'test_1_llama_32_predictions_few_fold1_True.csv', 
                      'test_2_llama_32_predictions_few_fold2_False.csv', 
                      'test_2_llama_32_predictions_few_fold2_True.csv']
    for file in files_to_check:
        file_path = os.path.join("files", file)
        if not os.path.exists(file_path):
            print(f"Archivo no encontrado: {file_path}")
            continue
        
        print(f"Archivo encontrado: {file_path}")
        model_name = os.path.splitext(file)[0]
        results = evaluate_gemma3(file_path, model_name=model_name, positive_label="anorexia")
        df = pd.DataFrame([results])
        print(df.to_string(index=False))

if __name__ == "__main__":
    # main()
    print("Verificando archivos y evaluando métricas...")
    check_files()
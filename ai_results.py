import argparse
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def load_gemma3_predictions(csv_path):
    df = pd.read_csv(csv_path)
    expected = {"real_class", "new_class"}
    if not expected.issubset(df.columns):
        raise ValueError(
            f"El archivo {csv_path} debe contener las columnas {expected}. "
            f"Columnas actuales: {list(df.columns)}"
        )
    return df["real_class"].astype(str), df["new_class"].astype(str)


def compute_specificity(y_true, y_pred, positive_label=None):
    labels = sorted(set(y_true) | set(y_pred))
    if len(labels) != 2:
        raise ValueError("La especificidad solo se calcula para clasificador binario.")

    if positive_label is None:
        positive_label = labels[0]

    negative_label = [label for label in labels if label != positive_label][0]
    cm = confusion_matrix(y_true, y_pred, labels=[positive_label, negative_label])
    tp, fn, fp, tn = cm.ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate_gemma3(csv_path, model_name="gemma3", positive_label="anorexia"):
    y_true, y_pred = load_gemma3_predictions(csv_path)

    results = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "specificity": compute_specificity(y_true, y_pred, positive_label=positive_label),
    }

    return results


def main():
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

    results = evaluate_gemma3(args.csv_path, model_name="gemma3", positive_label=args.positive_label)
    df = pd.DataFrame([results])

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

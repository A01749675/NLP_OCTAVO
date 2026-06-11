"""Evaluation of fine-tuned RoBERTuito models on blind test datasets.

This script loads a locally fine-tuned RoBERTuito classification model
and its tokenizer, preprocesses a set of test tweets using pysentimiento,
and evaluates the model's performance by computing various classification
metrics including accuracy, precision, recall, F1, specificity, and AUC.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from torch.utils.data import Dataset
from datetime import datetime
import os
from pysentimiento.preprocessing import preprocess_tweet


# =====================================================================
# 1. Definición de la clase Dataset
# =====================================================================
class TweetDataset(Dataset):
    """PyTorch Dataset wrapper for tokenized tweets and their labels.

    Parameters
    ----------
    encodings : dict
        A dictionary containing the tokenized inputs (e.g., input_ids,
        attention_mask) returned by a Hugging Face tokenizer.
    labels : list or array-like
        The numerical class labels corresponding to the tokenized inputs.
    """

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item


# =====================================================================
# 2. Definir métricas de evaluación
# =====================================================================
def compute_metrics(pred):
    """Calculate evaluation metrics for the model's predictions.

    Parameters
    ----------
    pred : transformers.EvalPrediction
        An object containing the model's predictions (logits) and the
        true label ids.

    Returns
    -------
    dict
        A dictionary containing the computed metrics: accuracy, f1,
        precision, recall, specificity, and auc.
    """
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)

    # Aplicar Softmax para obtener las probabilidades de la clase positiva (anorexia)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)

    # Calcular la especificidad (Tasa de Verdaderos Negativos)
    specificity = recall_score(labels, preds, pos_label=0)

    # Calcular las métricas estándar de clasificación
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auc
    }


# =====================================================================
# 3. Función principal de evaluación
# =====================================================================
def evaluate_test_data_robertuito(
        input_csv="files/data_test_fold2(in).csv",
        model_dir="./modelo_robertuito_final"
):
    """Evaluate the fine-tuned RoBERTuito model on a blind test dataset.

    Parameters
    ----------
    input_csv : str, optional
        Path to the CSV file containing the test data. Defaults to
        "files/data_test_fold2(in).csv".
    model_dir : str, optional
        Directory containing the fine-tuned Hugging Face model and
        tokenizer. Defaults to "./modelo_robertuito_final".

    Returns
    -------
    dict
        A dictionary containing the evaluation results computed by the Trainer.

    Raises
    ------
    RuntimeError
        If the model or tokenizer cannot be loaded from the specified directory.
    FileNotFoundError
        If the specified input CSV file does not exist.
    ValueError
        If the test dataset is empty after filtering out invalid labels.
    """
    print(f"Cargando tokenizador y modelo desde: {model_dir}")

    # Inicializar el tokenizador y el modelo de clasificación de secuencias
    try:
        tokenizer_test = AutoTokenizer.from_pretrained(model_dir)
        modelo_test = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        raise RuntimeError(
            f"Error al cargar el modelo de la carpeta '{model_dir}'. ¿Estás seguro de que existe y contiene los archivos del modelo? Error: {e}")

    print(f"Cargando datos de prueba desde: {input_csv}")

    # Leer el archivo CSV con los datos de prueba
    try:
        df = pd.read_csv(input_csv, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"No se pudo encontrar el archivo '{input_csv}'.")

    # Limpieza preventiva de clases para estandarizar el texto y evitar NaNs
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()

    mapeo_clases = {
        "control": 0,
        "anorexia": 1
    }

    # Filtrar posibles registros con etiquetas inválidas que no pertenezcan a la taxonomía definida
    clases_validas = df["class_clean"].isin(mapeo_clases.keys())
    if not clases_validas.all():
        invalid_count = (~clases_validas).sum()
        print(f"¡ADVERTENCIA! Filtrando {invalid_count} registros con etiquetas desconocidas en el test set.")
        df = df[clases_validas].copy()

    # Mapear las clases de texto a identificadores numéricos
    df["label_id"] = df["class_clean"].map(mapeo_clases)

    # Extraer textos y etiquetas en listas independientes
    X = df["tweet_text"].fillna("").astype(str).tolist()
    y = df["label_id"].tolist()

    if not X:
        raise ValueError("El dataset de prueba está vacío tras el filtrado.")

    print(f"Total de muestras válidas para test: {len(X)}")

    # -----------------------------------------------------------------
    # LA MAGIA DE ROBERTUITO: Preprocesamiento nativo de pysentimiento
    # -----------------------------------------------------------------
    print("Traduciendo emojis y preprocesando jerga de Twitter...")
    X_preprocesado = [preprocess_tweet(tweet) for tweet in X]

    # Aplicar la tokenización con padding dinámico y truncamiento
    print("Tokenizando textos...")
    test_encodings = tokenizer_test(X_preprocesado, padding=True, truncation=True, max_length=128)

    # Instanciar el objeto Dataset de PyTorch para el conjunto de prueba
    dataset = TweetDataset(test_encodings, y)

    # Inicializar el objeto Trainer optimizado para el cálculo de inferencia y métricas
    evaluador = Trainer(
        model=modelo_test,
        eval_dataset=dataset,
        compute_metrics=compute_metrics
    )

    # Ejecutar la evaluación global del modelo sobre los datos de prueba
    print("\nIniciando evaluación en los nuevos datos de test...")
    resultados = evaluador.evaluate()

    print("\n--- Resultados de RoBERTuito Test ---")
    for key, value in resultados.items():
        print(f"{key}: {value}")

    # Guardar un reporte de texto en disco con las métricas detalladas y la marca de tiempo
    nombre_archivo = "metricas_evaluacion_test_robertuito.txt"
    ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(nombre_archivo, "w", encoding="utf-8") as f:
        f.write("==================================================\n")
        f.write(f"REPORTE DE EVALUACIÓN TEST CIEGO - ROBERTUITO\n")
        f.write(f"Fecha y Hora: {ahora}\n")
        f.write("==================================================\n\n")

        for metrica, valor in resultados.items():
            if isinstance(valor, float) and "time" not in metrica and "per_second" not in metrica:
                f.write(f"{metrica:<30}: {valor:.4f}\n")
            else:
                f.write(f"{metrica:<30}: {valor}\n")

        f.write("\n==================================================\n")

    print(f"¡Métricas guardadas con éxito en '{nombre_archivo}'!")

    return resultados


if __name__ == "__main__":
    # Ajustar las rutas relativas si la estructura del proyecto cambia
    evaluate_test_data_robertuito(
        input_csv="files/data_test_fold2(in).csv",  # Usa tu archivo de test limpio
        model_dir="./modelo_robertuito_final"  # La carpeta donde se guardó el entrenamiento
    )
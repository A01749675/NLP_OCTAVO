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
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)

    # Softmax para probabilidades de la clase positiva (anorexia)
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)

    # Specificity
    specificity = recall_score(labels, preds, pos_label=0)

    # Métricas estándar
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
    """
    Evalúa el modelo RoBERTuito afinado en un dataset de prueba ciego.
    """
    print(f"Cargando tokenizador y modelo desde: {model_dir}")

    # Usar AutoTokenizer y AutoModelForSequenceClassification para RoBERTa
    try:
        tokenizer_test = AutoTokenizer.from_pretrained(model_dir)
        modelo_test = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        raise RuntimeError(
            f"Error al cargar el modelo de la carpeta '{model_dir}'. ¿Estás seguro de que existe y contiene los archivos del modelo? Error: {e}")

    print(f"Cargando datos de prueba desde: {input_csv}")
    try:
        df = pd.read_csv(input_csv, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"No se pudo encontrar el archivo '{input_csv}'.")

    # Limpieza preventiva de clases para evitar NaNs en las etiquetas
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()

    mapeo_clases = {
        "control": 0,
        "anorexia": 1
    }

    # Filtrar posibles registros con etiquetas inválidas
    clases_validas = df["class_clean"].isin(mapeo_clases.keys())
    if not clases_validas.all():
        invalid_count = (~clases_validas).sum()
        print(f"¡ADVERTENCIA! Filtrando {invalid_count} registros con etiquetas desconocidas en el test set.")
        df = df[clases_validas].copy()

    df["label_id"] = df["class_clean"].map(mapeo_clases)

    # Extraer textos y etiquetas
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

    # Tokenización
    print("Tokenizando textos...")
    test_encodings = tokenizer_test(X_preprocesado, padding=True, truncation=True, max_length=128)

    # Crear Dataset
    dataset = TweetDataset(test_encodings, y)

    # Inicializar Evaluador (Trainer de inferencia)
    evaluador = Trainer(
        model=modelo_test,
        eval_dataset=dataset,
        compute_metrics=compute_metrics
    )

    # Evaluar
    print("\nIniciando evaluación en los nuevos datos de test...")
    resultados = evaluador.evaluate()

    print("\n--- Resultados de RoBERTuito Test ---")
    for key, value in resultados.items():
        print(f"{key}: {value}")

    # Guardar reporte
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
    # Ajusta los paths si tu estructura de carpetas es diferente
    evaluate_test_data_robertuito(
        input_csv="files/data_test_fold2(in).csv",  # Usa tu archivo de test limpio
        model_dir="./modelo_robertuito_final"  # La carpeta donde se guardó el entrenamiento
    )
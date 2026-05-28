import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from torch.utils.data import Dataset
from text_cleaner import process_csv2
from datetime import datetime


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

process_csv2('data_test_fold1(in).csv', 'data_test_clean2.csv','tweet_text')

# Cargar el modelo
ruta_modelo = "./modelo_beto_final"
tokenizer_test = BertTokenizer.from_pretrained(ruta_modelo)
modelo_test = BertForSequenceClassification.from_pretrained(ruta_modelo)

# Carga y preparación de datos (Asegúrate de que el nombre del archivo termine en .csv si aplica)
df = pd.read_csv("files/data_test_clean2.csv", encoding="utf-8") # Añadí .csv por si acaso

X = df["tweet_text_clean"]
y = df["class"]

if X.empty:
    raise ValueError("No data")
else:
    print("Datos cargados correctamente. Listo para procesar.")

# Preparar textos y etiquetas
texts = X.fillna("").astype(str).tolist()
# --- CORRECCIÓN DEL MAPEO DE CLASES ---
# 1. Limpiamos espacios invisibles y pasamos a minúsculas para evitar errores de tipeo
y_limpio = y.astype(str).str.strip().str.lower()

# 2. Forzamos el diccionario para que no cambie jamás
mapeo_clases = {
    "control": 0,
    "anorexia": 1
}

# 3. Aplicamos el mapa y lo convertimos a lista
label_ids = y_limpio.map(mapeo_clases).tolist()

# Pequeña validación para asegurarnos de que no haya nulos por palabras mal escritas
if pd.Series(label_ids).isna().any():
    print("¡ADVERTENCIA! Hay palabras en tu columna 'class' que no son ni 'anorexia' ni 'control'.")


test_encodings = tokenizer_test(texts, padding=True, truncation=True, max_length=128)


dataset = TweetDataset(test_encodings, label_ids)


def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)
    
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)
    specificity = recall_score(labels, preds, pos_label=0)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'auc': auc
    }


evaluador = Trainer(
    model=modelo_test,
    eval_dataset=dataset,
    compute_metrics=compute_metrics
)

# Evaluar
print("Iniciando evaluación en los nuevos datos de test...")
resultados = evaluador.evaluate()

print("\n--- Resultados del Test ---")
print(resultados)

nombre_archivo = "metricas_evaluacion_test.txt"

# Obtenemos la fecha y hora actual para saber cuándo hicimos la prueba
ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(nombre_archivo, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write(f"REPORTE DE EVALUACIÓN - BETO\n")
    f.write(f"Fecha y Hora: {ahora}\n")
    f.write("==================================================\n\n")
    
    # Recorremos el diccionario de resultados y lo escribimos bonito
    for metrica, valor in resultados.items():
        # Formateamos las métricas flotantes para que no tengan tantos decimales
        if isinstance(valor, float) and "time" not in metrica and "per_second" not in metrica:
            f.write(f"{metrica:<30}: {valor:.4f}\n")
        else:
            f.write(f"{metrica:<30}: {valor}\n")
            
    f.write("\n==================================================\n")

print(f"¡Métricas guardadas con éxito en '{nombre_archivo}'!")


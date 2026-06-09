import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from torch.utils.data import Dataset
from pysentimiento.preprocessing import preprocess_tweet
from datetime import datetime
import os

from text_cleaner import text_filtering2  # Asegúrate de tener esta función definida en tu proyecto

# =====================================================================
# 1. Definición de la clase Dataset (Igual para PyTorch)
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
# 2. Cargar Tokenizer de RoBERTuito (Usa AutoTokenizer para RoBERTa)
# =====================================================================
# robertuito-sentiment-analysis maneja emojis y jerga de Twitter nativamente
nombre_modelo = "pysentimiento/robertuito-base-cased"
tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)

# =====================================================================
# 3. Carga y preparación de datos
# =====================================================================
df = pd.read_csv("files/data_train(in).csv", encoding="utf-8")

if df.empty:
    raise ValueError("El archivo CSV está vacío.")

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
    print(f"¡ADVERTENCIA! Filtrando {invalid_count} registros con etiquetas desconocidas.")
    df = df[clases_validas].copy()

df["label_id"] = df["class_clean"].map(mapeo_clases)

X = df["tweet_text"].fillna("").astype(str).tolist()
y = df["label_id"].tolist()

print(f"Datos cargados correctamente para RoBERTuito. Total muestras válidas: {len(X)}")

# --- Split de Entrenamiento / Validación (70/30) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

print("Traduciendo emojis y preprocesando jerga de Twitter...")
X_train_pre = [text_filtering2(tweet) for tweet in X_train]
X_test_pre = [text_filtering2(tweet) for tweet in X_test]

train_encodings = tokenizer(X_train_pre, padding=True, truncation=True, max_length=128)
test_encodings = tokenizer(X_test_pre, padding=True, truncation=True, max_length=128)

train_dataset = TweetDataset(train_encodings, y_train)
test_dataset = TweetDataset(test_encodings, y_test)

# =====================================================================
# 5. Cargar RoBERTuito preparado para Clasificación Binaria
# =====================================================================
num_clases = 2 

# ignore_mismatched_sizes=True es indispensable aquí.
# Esto le dice al script: "Borra la cabeza original de 3 clases (positivo/negativo/neutral) 
# de análisis de sentimiento y monta una nueva cabeza limpia para nuestras 2 clases".
modelo = AutoModelForSequenceClassification.from_pretrained(
    nombre_modelo, 
    num_labels=num_clases,
    ignore_mismatched_sizes=True 
)

# Dejamos todas las capas abiertas para el ajuste fino inicial
for name, param in modelo.roberta.named_parameters():
    param.requires_grad = True

print("Estructura de RoBERTuito inicializada y adaptada para 2 clases.")

# =====================================================================
# 6. Definir métricas de evaluación
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
# 7. Definir Hiperparámetros Optimizados para Muestras Pequeñas (3,000 tweets)
# =====================================================================
training_args = TrainingArguments(
    output_dir='./resultados_robertuito',          
    num_train_epochs=5,                        # 5 epochs le da suficiente espacio al Early Stopping
    per_device_train_batch_size=16,            # Batch de 16 para mayor frecuencia de actualización en datos chicos
    per_device_eval_batch_size=32,           
    learning_rate=2e-5,
    weight_decay=0.05,                         # Regularización moderada para evitar memorizar palabras clave ruidosas
    warmup_ratio=0.10,
    lr_scheduler_type="linear",                # Decaimiento lineal estable para pocos steps por epoch
    eval_strategy="epoch",                   
    save_strategy="epoch",                    
    load_best_model_at_end=True,               # Revierte automáticamente al checkpoint óptimo
    metric_for_best_model="eval_loss",                # Buscamos maximizar el F1-Score macro/general
    greater_is_better=False,
    save_total_limit=1,                        # Mantiene solo el mejor modelo para ahorrar espacio
    disable_tqdm=False,
    report_to="none",
    logging_steps=10,
)

# =====================================================================
# 8. Inicializar el Trainer con Parada Temprana (Early Stopping)
# =====================================================================
trainer = Trainer(
    model=modelo,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] # Frena si hila 2 epochs empeorando
)

print("Iniciando el entrenamiento de RoBERTuito...")

# =====================================================================
# 9. Entrenar el modelo
# =====================================================================
trainer.train()

# =====================================================================
# 10. Evaluar en el conjunto de validación
# =====================================================================
print("Evaluando el mejor modelo encontrado de RoBERTuito...")
resultados = trainer.evaluate()
print("Resultados de la evaluación:", resultados)

# =====================================================================
# 11. Guardar el mejor modelo definitivo y el tokenizador
# =====================================================================
ruta_guardado = "./modelo_robertuito_final"
print(f"Guardando el modelo definitivo y el tokenizador en: {ruta_guardado}")

trainer.save_model(ruta_guardado)
tokenizer.save_pretrained(ruta_guardado)

# --- GUARDAR MÉTRICAS EN UN TXT ---
nombre_archivo = "metricas_evaluacion_robertuito2.txt"
ahora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

with open(nombre_archivo, "w", encoding="utf-8") as f:
    f.write("==================================================\n")
    f.write(f"REPORTE DE EVALUACIÓN - ROBERTUITO FINETUNED\n")
    f.write(f"Fecha y Hora: {ahora}\n")
    f.write("==================================================\n\n")
    
    for metrica, valor in resultados.items():
        if isinstance(valor, float) and "time" not in metrica and "per_second" not in metrica:
            f.write(f"{metrica:<30}: {valor:.4f}\n")
        else:
            f.write(f"{metrica:<30}: {valor}\n")
            
    f.write("\n==================================================\n")

print(f"¡Métricas guardadas con éxito en '{nombre_archivo}'!")
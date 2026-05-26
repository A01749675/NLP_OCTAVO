import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from torch.utils.data import Dataset
from datetime import datetime

# 1. Definición de la clase Dataset
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

# 2. Cargar Tokenizer
tokenizer = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# 3. Carga y preparación de datos
df = pd.read_csv("files/data_train_cleaned2.csv", encoding="utf-8")

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

# --- Entrenamiento 70/30 ---
X_train, X_test, y_train, y_test = train_test_split(
    texts, label_ids, test_size=0.3, random_state=42, stratify=label_ids
)

# Tokenizar por separado para cada split
train_encodings = tokenizer(X_train, padding=True, truncation=True, max_length=128)
test_encodings = tokenizer(X_test, padding=True, truncation=True, max_length=128)

# 4. Crear los objetos Dataset para PyTorch
train_dataset = TweetDataset(train_encodings, y_train)
test_dataset = TweetDataset(test_encodings, y_test)

# 5. Cargar BETO preparado para clasificación de secuencias
num_clases = 2 
modelo = BertForSequenceClassification.from_pretrained(
    "dccuchile/bert-base-spanish-wwm-cased", 
    num_labels=num_clases
)

# 6. Definir métricas de evaluación (ACTUALIZADO CON SPECIFICITY Y AUC)
def compute_metrics(pred):
    labels = pred.label_ids
    logits = pred.predictions
    preds = logits.argmax(-1)
    
    # 1. Calcular AUC:
    # El modelo nos devuelve "logits" (números crudos). Los convertimos a probabilidades 
    # de 0 a 1 usando la función Softmax, y extraemos solo las probabilidades de la clase 1.
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    auc = roc_auc_score(labels, probs)
    
    # 2. Calcular Specificity (Especificidad):
    # La especificidad equivale matemáticamente al Recall de la clase negativa (0).
    specificity = recall_score(labels, preds, pos_label=0)
    
    # 3. Métricas estándar
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

# 7. Definir Hiperparámetros OPTIMIZADOS (TrainingArguments)
training_args = TrainingArguments(
    output_dir='./resultados_beto',          
    num_train_epochs=4,                      
    per_device_train_batch_size=16,          
    per_device_eval_batch_size=32,           
    learning_rate=1e-5,                      
    weight_decay=0.01,                       
    eval_strategy="epoch",                   # Evaluar al final de cada epoch para monitorear métricas
    save_strategy="no",                      # Evita que se guarden archivos pesados por cada epoch
    load_best_model_at_end=False,            # Requerido al desactivar el guardado
)

# 8. Inicializar el Trainer
trainer = Trainer(
    model=modelo,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Iniciando el entrenamiento")

# 9. Entrenar el modelo
trainer.train()

# 10. Evaluar en el conjunto de prueba
print("Evaluando el modelo")
resultados = trainer.evaluate()
print("Resultados de la evaluación:", resultados)

# 11. Guardar

print("Guardando el modelo entrenado y el tokenizador en disco...")
ruta_guardado = "./modelo_beto_final"

trainer.save_model(ruta_guardado)
tokenizer.save_pretrained(ruta_guardado)



# ... (Aquí va tu código anterior donde calculas 'resultados')

# --- NUEVO PASO: GUARDAR MÉTRICAS EN UN TXT ---
nombre_archivo = "metricas_evaluacion_entrenamiento.txt"

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



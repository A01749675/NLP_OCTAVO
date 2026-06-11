"""Batch inference and evaluation for eating disorder classification.

This script loads a fine-tuned Llama 3.2 model via Unsloth, processes a
blind test dataset of tweets, performs batch generation to predict
binary classes ('anorexia' vs 'control'), and computes standard
classification metrics to evaluate model performance.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch


def main():
    """Execute the batch inference and evaluation pipeline.

    Loads the optimized model and tokenizer, prepares the test dataset,
    formats prompts for classification, runs batch inference on a GPU,
    and prints the final evaluation metrics.

    Returns
    -------
    None
    """
    print("1. Cargando el modelo entrenado...")

    # Cargar el modelo cuantizado (4-bit) y el tokenizador usando Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="llama3.2_clasificador_tca",
        max_seq_length=256,
        load_in_4bit=True,
    )

    # Activar optimizaciones de inferencia (aumento de velocidad) nativas de Unsloth
    FastLanguageModel.for_inference(model)

    # Configurar el padding a la izquierda, requisito indispensable para la inferencia por lotes (batching) en Llama
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("2. Cargando el Test Set ciego (100% de los datos)...")

    # Cargar el conjunto de datos de prueba desde un archivo local
    df = pd.read_csv("files/data_test_fold1(in).csv", encoding="utf-8")

    # Limpiar valores nulos y mapear las etiquetas de texto a identificadores numéricos binarios
    df = df.dropna(subset=["tweet_text", "class"])
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()
    df["label_id"] = df["class_clean"].map({"control": 0, "anorexia": 1})

    test_df = df

    # Definir la plantilla del prompt instruccional forzando una respuesta de una sola palabra
    prompt_template_test = """Eres un experto psicólogo clínico. Clasifica el siguiente tweet estrictamente como 'anorexia' o 'control'. Responde solo con esa palabra.

### Tweet:
{}

### Clase:
"""

    y_true = test_df["label_id"].tolist()
    y_pred_classes = []
    y_pred_probs = []

    prompts = [prompt_template_test.format(text) for text in test_df["tweet_text"]]
    batch_size = 8  # Evaluará de 8 en 8 tweets a la vez

    print("\n3. Iniciando Clasificación (Batching)...")

    # Iterar sobre la lista de prompts en fragmentos definidos por batch_size para optimizar VRAM
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenizar el lote de prompts asegurando padding dinámico, truncamiento y envío del tensor a la GPU
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=256).to("cuda")

        # Desactivar el cálculo de gradientes para reducir drásticamente el consumo de memoria durante inferencia
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # SUFICIENTES TOKENS PARA LA PALABRA COMPLETA
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Procesar secuencialmente las secuencias generadas en el lote actual
        for j, output in enumerate(outputs):
            # Extraer y decodificar únicamente los nuevos tokens generados, omitiendo el contexto del prompt original
            respuesta = tokenizer.decode(output[inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True).strip().lower()

            # Clasificar heurísticamente basado en la presencia de la palabra clave en la respuesta decodificada
            if "anorexia" in respuesta:
                y_pred_classes.append(1)
                y_pred_probs.append(0.99)
            else:
                y_pred_classes.append(0)
                y_pred_probs.append(0.01)

        # Liberar explícitamente los tensores del lote actual y limpiar la caché de CUDA para evitar desbordamientos
        del inputs, outputs
        torch.cuda.empty_cache()

    print("\n=== RESULTADOS FINALES TEST CIEGO ===")

    # Calcular y reportar las métricas estadísticas para evaluar la calidad de la clasificación
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='binary',
                                                               zero_division=0)
    acc = accuracy_score(y_true, y_pred_classes)
    specificity = recall_score(y_true, y_pred_classes, pos_label=0)
    auc = roc_auc_score(y_true, y_pred_probs)

    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 Score:    {f1:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"AUC:         {auc:.4f}")


if __name__ == "__main__":
    main()
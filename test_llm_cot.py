"""Batch inference and evaluation for Chain-of-Thought (CoT) models.

This script loads a fine-tuned Llama 3.2 model optimized with Unsloth,
processes a test dataset, and generates step-by-step clinical reasoning
followed by a classification. It logs the generated reasoning and evaluates
the model's performance using standard classification metrics.
"""

import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from unsloth import FastLanguageModel
from tqdm import tqdm


def main():
    """Execute the CoT inference and evaluation pipeline.

    Loads the optimized model, prepares the dataset, runs batch generation
    to extract both reasoning and final classification, saves detailed logs,
    and computes aggregate evaluation metrics.

    Returns
    -------
    None
    """
    # =====================================================================
    # 1. Configuración y Carga del Modelo
    # =====================================================================
    model_path = "llama3.2_clasificador_tca_cot"  # Tu modelo entrenado con CoT
    output_log_file = "files/resultados_cot_logs.csv"

    print(f"1. Cargando el modelo CoT desde: {model_path}...")

    # Cargar el modelo en 4-bit con Unsloth asegurando el tamaño máximo de secuencia usado en entrenamiento
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,  # Coherente con tu entrenamiento
        load_in_4bit=True,
    )

    # Activar optimizaciones de inferencia (aumento de velocidad) nativas de Unsloth
    FastLanguageModel.for_inference(model)

    # Configurar padding a la izquierda, requisito indispensable para inferencia por lotes (batching) en Llama
    # CRÍTICO PARA INFERENCIA POR LOTES
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================================
    # 2. Carga de Datos
    # =====================================================================
    print("2. Cargando el Test Set ciego...")

    # Leer el dataset de prueba y procesar las etiquetas verdaderas
    df = pd.read_csv("files/data_test_fold1(in).csv", encoding="utf-8")
    df = df.dropna(subset=["tweet_text", "class"])
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()
    df["label_id"] = df["class_clean"].map({"control": 0, "anorexia": 1})

    # Definir la plantilla del prompt instruccional forzando un análisis paso a paso antes de clasificar
    prompt_template_test = """Eres un experto psicólogo clínico. Analiza el siguiente tweet paso a paso evaluando posibles indicadores, y luego clasifícalo estrictamente como 'anorexia' o 'control'.

### Tweet:
{}

### Análisis:
"""

    prompts = [prompt_template_test.format(text) for text in df["tweet_text"]]

    # El CoT genera secuencias largas, se ajusta el batch_size para evitar desbordamiento de VRAM
    batch_size = 8

    y_true = df["label_id"].tolist()
    y_pred_classes = []
    y_pred_probs = []

    # Lista para almacenar los resultados detallados
    logs_data = []

    # =====================================================================
    # 3. Inferencia y Extracción
    # =====================================================================
    print(f"\n3. Iniciando Clasificación (Batch size: {batch_size})...")

    # Iterar sobre la lista de prompts en fragmentos (batches)
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_tweets = df["tweet_text"].iloc[i:i + batch_size].tolist()
        batch_real_classes = df["class_clean"].iloc[i:i + batch_size].tolist()

        # Tokenizar el lote de entradas con padding dinámico y enviarlo a la GPU
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        # Desactivar el cálculo de gradientes para optimizar el consumo de memoria durante la generación
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # ESPACIO PARA EL RAZONAMIENTO COMPLETO
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Procesar secuencialmente cada salida generada en el lote actual
        for j, output in enumerate(outputs):
            # 1. Decodificar solo la respuesta generada omitiendo el prompt original
            respuesta_cruda = tokenizer.decode(output[inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True).strip().lower()

            # 2. Separar el bloque de razonamiento de la clase final usando la etiqueta delimitadora
            if "### clase final:" in respuesta_cruda:
                partes = respuesta_cruda.split("### clase final:")
                razonamiento = partes[0].strip()
                clase_predicha = partes[-1].strip()
            else:
                # Fallback por si el modelo omite el formato de delimitación esperado
                razonamiento = respuesta_cruda
                clase_predicha = respuesta_cruda

            # 3. Mapeo probabilístico y numérico basado en la presencia de la palabra clave
            if "anorexia" in clase_predicha:
                pred_id = 1
                prob = 0.99
            else:
                pred_id = 0
                prob = 0.01

            y_pred_classes.append(pred_id)
            y_pred_probs.append(prob)

            # 4. Guardar en el log los metadatos y el texto generado para auditoría
            logs_data.append({
                "tweet_text": batch_tweets[j],
                "clase_real": batch_real_classes[j],
                "clase_predicha": "anorexia" if pred_id == 1 else "control",
                "razonamiento_generado": razonamiento,
                "acierto": 1 if (batch_real_classes[j] == ("anorexia" if pred_id == 1 else "control")) else 0
            })

        # Liberar explícitamente la memoria de los tensores procesados y limpiar caché
        del inputs, outputs
        torch.cuda.empty_cache()

        # Guardado de seguridad parcial en disco (checkpointing)
        if i > 0 and i % 50 == 0:
            pd.DataFrame(logs_data).to_csv(output_log_file, index=False, encoding="utf-8")

    # =====================================================================
    # 4. Resultados Finales y Guardado
    # =====================================================================
    # Guardado del CSV final completo con todos los registros
    logs_df = pd.DataFrame(logs_data)
    logs_df.to_csv(output_log_file, index=False, encoding="utf-8")
    print(f"\n[OK] Log de inferencia guardado exitosamente en: {output_log_file}")

    print("\n=== RESULTADOS FINALES TEST CIEGO (CoT) ===")

    # Calcular y reportar las métricas de clasificación consolidadas
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
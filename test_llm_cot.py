import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from unsloth import FastLanguageModel
from tqdm import tqdm


def main():
    # =====================================================================
    # 1. Configuración y Carga del Modelo
    # =====================================================================
    model_path = "llama3.2_clasificador_tca_cot"  # Tu modelo entrenado con CoT
    output_log_file = "files/resultados_cot_logs.csv"

    print(f"1. Cargando el modelo CoT desde: {model_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,  # Coherente con tu entrenamiento
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # CRÍTICO PARA INFERENCIA POR LOTES
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =====================================================================
    # 2. Carga de Datos
    # =====================================================================
    print("2. Cargando el Test Set ciego...")
    df = pd.read_csv("files/data_test_fold1(in).csv", encoding="utf-8")
    df = df.dropna(subset=["tweet_text", "class"])
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()
    df["label_id"] = df["class_clean"].map({"control": 0, "anorexia": 1})

    prompt_template_test = """Eres un experto psicólogo clínico. Analiza el siguiente tweet paso a paso evaluando posibles indicadores, y luego clasifícalo estrictamente como 'anorexia' o 'control'.

### Tweet:
{}

### Análisis:
"""

    prompts = [prompt_template_test.format(text) for text in df["tweet_text"]]

    # El CoT genera secuencias largas
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

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        batch_tweets = df["tweet_text"].iloc[i:i + batch_size].tolist()
        batch_real_classes = df["class_clean"].iloc[i:i + batch_size].tolist()

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,  # ESPACIO PARA EL RAZONAMIENTO COMPLETO
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        for j, output in enumerate(outputs):
            # 1. Decodificar solo la respuesta generada
            respuesta_cruda = tokenizer.decode(output[inputs["input_ids"].shape[1]:],
                                               skip_special_tokens=True).strip().lower()

            # 2. Separar el razonamiento de la clase final
            if "### clase final:" in respuesta_cruda:
                partes = respuesta_cruda.split("### clase final:")
                razonamiento = partes[0].strip()
                clase_predicha = partes[-1].strip()
            else:
                # Fallback por si el modelo omite el formato
                razonamiento = respuesta_cruda
                clase_predicha = respuesta_cruda

                # 3. Mapeo probabilístico y numérico
            if "anorexia" in clase_predicha:
                pred_id = 1
                prob = 0.99
            else:
                pred_id = 0
                prob = 0.01

            y_pred_classes.append(pred_id)
            y_pred_probs.append(prob)

            # 4. Guardar en el log
            logs_data.append({
                "tweet_text": batch_tweets[j],
                "clase_real": batch_real_classes[j],
                "clase_predicha": "anorexia" if pred_id == 1 else "control",
                "razonamiento_generado": razonamiento,
                "acierto": 1 if (batch_real_classes[j] == ("anorexia" if pred_id == 1 else "control")) else 0
            })

        del inputs, outputs
        torch.cuda.empty_cache()

        # Guardado de seguridad parcial (opcional pero recomendado)
        if i > 0 and i % 50 == 0:
            pd.DataFrame(logs_data).to_csv(output_log_file, index=False, encoding="utf-8")

    # =====================================================================
    # 4. Resultados Finales y Guardado
    # =====================================================================
    # Guardado del CSV final completo
    logs_df = pd.DataFrame(logs_data)
    logs_df.to_csv(output_log_file, index=False, encoding="utf-8")
    print(f"\n[OK] Log de inferencia guardado exitosamente en: {output_log_file}")

    print("\n=== RESULTADOS FINALES TEST CIEGO (CoT) ===")
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
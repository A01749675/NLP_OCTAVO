"""Fine-tuning of Llama 3.2 using QLoRA and Chain-of-Thought (CoT).

This script configures a QLoRA pipeline optimized for complex reasoning.
It loads a 4-bit quantized Llama 3.2 model, injects higher-rank LoRA
adapters to capture logical steps, and trains the model on a dataset
containing step-by-step clinical analyses (Chain of Thought) before
outputting a final classification.
"""

import os
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from tqdm import tqdm

# Parche de seguridad para prevenir colisiones en librerías OpenMP de Intel/Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    """Execute the QLoRA fine-tuning pipeline with Chain-of-Thought (CoT).

    Handles the end-to-end process: loading the quantized base model, configuring
    higher-capacity LoRA adapters, formatting prompts to induce step-by-step
    reasoning, training with gradient accumulation to manage larger sequence
    lengths, evaluating logical generation, and saving the fine-tuned adapters.

    Returns
    -------
    None
    """
    # =====================================================================
    # 1. Configuración de Arquitectura y VRAM (Ajustado para CoT)
    # =====================================================================
    max_seq_length = 512
    load_in_4bit = True

    print("Cargando Llama 3.2 (3B) en 4-bits...")

    # Cargar el modelo base y el tokenizador en 4-bits
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    # Inyectar adaptadores LoRA. Se incrementa el rango (r=32, lora_alpha=32)
    # para dotar al modelo de mayor capacidad paramétrica para aprender razonamiento lógico.
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Optimización extra de memoria nativa de Unsloth
        random_state=42,
    )

    # =====================================================================
    # 2. Preparación de Datos con Chain of Thought
    # =====================================================================
    # El prompt de entrenamiento ahora exige y estructura el análisis antes de la clase final
    prompt_template_train = """Eres un experto psicólogo clínico. Analiza el siguiente tweet paso a paso evaluando posibles indicadores, y luego clasifícalo estrictamente como 'anorexia' o 'control'.

### Tweet:
{}

### Análisis:
{}

### Clase Final:
{}"""

    # El prompt de inferencia deja vacía la sección de análisis para que el modelo la genere
    prompt_template_test = """Eres un experto psicólogo clínico. Analiza el siguiente tweet paso a paso evaluando posibles indicadores, y luego clasifícalo estrictamente como 'anorexia' o 'control'.

### Tweet:
{}

### Análisis:
"""

    print("Cargando y separando dataset con CoT...")

    # Cargar el nuevo dataset que contiene la columna de razonamiento generado previamente
    df = pd.read_csv("files/data_train_cot.csv", encoding="utf-8")
    df = df.dropna(subset=["tweet_text", "class", "reasoning"])
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()

    # Mapear las clases de texto a identificadores numéricos
    mapeo_clases = {"control": 0, "anorexia": 1}
    df["label_id"] = df["class_clean"].map(mapeo_clases)

    # Dividir el dataset preservando la proporción de clases (estratificación)
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label_id"])

    textos_train = []
    EOS_TOKEN = tokenizer.eos_token

    # Formatear cada ejemplo inyectando el Tweet, el Razonamiento y la Clase Final
    for _, row in train_df.iterrows():
        # Pasamos 3 variables: Tweet, Razonamiento y Clase
        texto = prompt_template_train.format(row["tweet_text"], row["reasoning"], row["class_clean"]) + EOS_TOKEN
        textos_train.append(texto)

    # Convertir a formato Dataset de Hugging Face
    dataset_train = Dataset.from_dict({"text": textos_train})

    # =====================================================================
    # 3. Entrenamiento (Hyperparámetros balanceados)
    # =====================================================================
    # Configurar SFTTrainer con estrategias de acumulación de gradientes para mitigar
    # el impacto en memoria de las secuencias más largas del CoT
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,  # CRÍTICO PARA WINDOWS: Evita bloqueos en multiprocessing
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=1,  # REDUCIDO: Para compensar secuencias más largas
            gradient_accumulation_steps=4,  # Incrementado para mantener un batch size efectivo de 4
            warmup_ratio=0.1,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            bf16=False,
            logging_steps=10,
            save_strategy="no",
            optim="paged_adamw_8bit",  # Optimizador paginado de 8 bits
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs_llama3.2_cot",
            report_to="none",
            disable_tqdm=False,
            torch_compile=False
        ),
    )

    print("\nIniciando Fine-Tuning con Chain of Thought...")
    trainer_stats = trainer.train()

    # =====================================================================
    # 4. Evaluación Lógica
    # =====================================================================
    print("\nEntrenamiento finalizado. Evaluando inferencia lógica...")

    # Activar optimizaciones de inferencia de Unsloth
    FastLanguageModel.for_inference(model)

    y_true = test_df["label_id"].tolist()
    y_pred_probs = []
    y_pred_classes = []

    # Iterar sobre el conjunto de prueba para generar predicciones
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Clasificando y razonando"):
        prompt = prompt_template_test.format(row["tweet_text"])
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        # Generar texto sin calcular gradientes
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decodificar la salida omitiendo el prompt original
        respuesta_completa = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:],
                                              skip_special_tokens=True).strip().lower()

        # Extraer solo la clase final del texto generado aislando el delimitador estructural
        if "### clase final:" in respuesta_completa:
            clase_predicha = respuesta_completa.split("### clase final:")[-1].strip()
        else:
            clase_predicha = respuesta_completa  # Fallback en caso de que el modelo pierda el formato

        # Mapeo probabilístico y numérico basado en la presencia de la palabra clave
        if "anorexia" in clase_predicha:
            y_pred_classes.append(1)
            y_pred_probs.append(0.99)
        else:
            y_pred_classes.append(0)
            y_pred_probs.append(0.01)

        # Liberar memoria VRAM residual en cada iteración
        del inputs, outputs
        torch.cuda.empty_cache()

    # Calcular y mostrar métricas de desempeño sobre la clase extraída del razonamiento
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='binary',
                                                               zero_division=0)
    acc = accuracy_score(y_true, y_pred_classes)
    specificity = recall_score(y_true, y_pred_classes, pos_label=0)
    auc = roc_auc_score(y_true, y_pred_probs)

    resultados = {
        'accuracy': acc, 'f1': f1, 'precision': precision,
        'recall': recall, 'specificity': specificity, 'auc': auc
    }

    print("\nResultados Llama 3.2 (Chain of Thought):")
    print(resultados)

    # =====================================================================
    # 5. Guardar Modelo
    # =====================================================================
    print("\nGuardando adaptadores LoRA...")

    # Guardar localmente los adaptadores entrenados sobre el espacio lógico de CoT
    model.save_pretrained("llama3.2_clasificador_tca_cot")
    tokenizer.save_pretrained("llama3.2_clasificador_tca_cot")
    print("¡Proceso 100% completado!")


# BARRERA CRÍTICA PARA WINDOWS: Impide colapsos de memoria al evitar ejecución paralela no deseada
if __name__ == "__main__":
    main()
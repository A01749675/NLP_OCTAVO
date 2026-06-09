import os
import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from datetime import datetime
from tqdm import tqdm

# Parche de seguridad para librerías de Intel/Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    # =====================================================================
    # 1. Configuración de Arquitectura Llama 3.2 y VRAM
    # =====================================================================
    max_seq_length = 256
    load_in_4bit = True

    print("Cargando Llama 3.2 (3B) en 4-bits...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
    )

    # =====================================================================
    # 2. Preparación de Datos
    # =====================================================================
    prompt_template_train = """Eres un experto psicólogo clínico. Clasifica el siguiente tweet estrictamente como 'anorexia' o 'control'. Responde solo con esa palabra.

### Tweet:
{}

### Clase:
{}"""

    prompt_template_test = """Eres un experto psicólogo clínico. Clasifica el siguiente tweet estrictamente como 'anorexia' o 'control'. Responde solo con esa palabra.

### Tweet:
{}

### Clase:
"""

    print("Cargando y separando dataset...")
    df = pd.read_csv("files/data_train(in).csv", encoding="utf-8")
    df = df.dropna(subset=["tweet_text", "class"])
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()

    mapeo_clases = {"control": 0, "anorexia": 1}
    df["label_id"] = df["class_clean"].map(mapeo_clases)

    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["label_id"])

    textos_train = []
    EOS_TOKEN = tokenizer.eos_token
    for _, row in train_df.iterrows():
        texto = prompt_template_train.format(row["tweet_text"], row["class_clean"]) + EOS_TOKEN
        textos_train.append(texto)

    dataset_train = Dataset.from_dict({"text": textos_train})

    # =====================================================================
    # 3. Entrenamiento
    # =====================================================================
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_train,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=1,  # CRÍTICO PARA WINDOWS: Forzar a 1 hilo
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            warmup_ratio=0.1,
            num_train_epochs=3,
            learning_rate=2e-4,
            fp16=True,
            bf16=False,
            logging_steps=10,
            save_strategy = "no",
            optim="paged_adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
            output_dir="outputs_llama3.2",
            report_to="none",
            disable_tqdm=False,
            torch_compile=False
        ),
    )

    print("\nIniciando Fine-Tuning...")
    trainer_stats = trainer.train()

    # =====================================================================
    # 4. Evaluación
    # =====================================================================
    print("\nEntrenamiento finalizado. Evaluando...")

    FastLanguageModel.for_inference(model)

    y_true = test_df["label_id"].tolist()
    y_pred_probs = []
    y_pred_classes = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Clasificando tweets"):
        prompt = prompt_template_test.format(row["tweet_text"])
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id  # Previene colapso en Llama
            )

        respuesta = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()

        if "anorexia" in respuesta:
            y_pred_classes.append(1)
            y_pred_probs.append(0.99)
        else:
            y_pred_classes.append(0)
            y_pred_probs.append(0.01)

        # Liberar memoria VRAM residual en cada iteración
        del inputs, outputs
        torch.cuda.empty_cache()

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='binary', zero_division=0)
    acc = accuracy_score(y_true, y_pred_classes)
    specificity = recall_score(y_true, y_pred_classes, pos_label=0)
    auc = roc_auc_score(y_true, y_pred_probs)

    resultados = {
        'accuracy': acc, 'f1': f1, 'precision': precision,
        'recall': recall, 'specificity': specificity, 'auc': auc
    }

    print("\nResultados Llama 3.2:")
    print(resultados)

    # =====================================================================
    # 5. Guardar Modelo (Hugging Face Format)
    # =====================================================================
    print("\nGuardando modelo y tokenizador...")

    # Esto guarda los adaptadores entrenados de forma instantánea
    model.save_pretrained("llama3.2_clasificador_tca")
    tokenizer.save_pretrained("llama3.2_clasificador_tca")

    print("¡Proceso 100% completado!")

# BARRERA CRÍTICA PARA WINDOWS
if __name__ == "__main__":
    main()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, recall_score
from unsloth import FastLanguageModel
from tqdm import tqdm
import torch


def main():
    print("1. Cargando el modelo entrenado...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="llama3.2_clasificador_tca",
        max_seq_length=256,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    # CRÍTICO PARA INFERENCIA POR LOTES: Llama necesita padding a la izquierda
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("2. Cargando el Test Set ciego (100% de los datos)...")
    # Carga tu nuevo archivo de prueba directamente
    df = pd.read_csv("files/data_test_fold1(in).csv", encoding="utf-8")

    df = df.dropna(subset=["tweet_text", "class"])
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()
    df["label_id"] = df["class_clean"].map({"control": 0, "anorexia": 1})

    test_df = df

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
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True,
            max_length=256).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=10,  # SUFICIENTES TOKENS PARA LA PALABRA COMPLETA
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )

        for j, output in enumerate(outputs):
            # Decodificar solo la respuesta nueva
            respuesta = tokenizer.decode(output[inputs["input_ids"].shape[1]:],
                                         skip_special_tokens=True).strip().lower()

            if "anorexia" in respuesta:
                y_pred_classes.append(1)
                y_pred_probs.append(0.99)
            else:
                y_pred_classes.append(0)
                y_pred_probs.append(0.01)

        del inputs, outputs
        torch.cuda.empty_cache()

    print("\n=== RESULTADOS FINALES TEST CIEGO ===")
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
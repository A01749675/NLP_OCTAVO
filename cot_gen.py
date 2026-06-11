import pandas as pd
from llama_cpp import Llama
from tqdm import tqdm
import os


def main():
    # =====================================================================
    # 1. Configuración de Rutas y Modelo
    # =====================================================================
    # Asegúrate de descargar la versión .gguf del modelo (ej. Q4_K_M)
    os.environ["LLAMA_ARG_N_CPU_MOE"] = "30" # 41 qwen | 30 gemma
    MODEL_PATH = "llms/models/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"

    input_file = "files/data_train(in).csv"
    output_file = "files/data_train_cot.csv"

    print(f"Cargando dataset desde {input_file}...")
    df = pd.read_csv(input_file, encoding="utf-8")
    df = df.dropna(subset=["tweet_text", "class"])
    df["class_clean"] = df["class"].astype(str).str.strip().str.lower()

    if "reasoning" not in df.columns:
        df["reasoning"] = ""

    # =====================================================================
    # 2. Inicialización de Llama.cpp
    # =====================================================================
    print(f"\nCargando modelo GGUF desde: {MODEL_PATH}")
    # Parámetros optimizados para rendimiento en GPU

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=31, # 999 qwen | 31 gemma
        n_ctx=2048,  # Ventana de contexto amplia para el prompt + razonamiento
        n_batch=512,  # Tamaño de lote para el procesamiento del prompt
        use_mmap=False,  # <--- EQUIVALENTE A --no-mmap
        verbose=False
    )

    system_prompt = """Eres un psicólogo clínico experto analizando el lenguaje en redes sociales.
Tu tarea es escribir el proceso de razonamiento paso a paso que explica por qué un tweet pertenece a una categoría específica ('anorexia' o 'control').

Debes usar ESTRICTAMENTE la siguiente estructura concisa y directa:
1. Observación: [Cita o describe los elementos clave del texto].
2. Implicación clínica: [Qué significa en el contexto de un TCA].
3. Conclusión: Por lo tanto, el texto se clasifica como [Clase].

No agregues saludos, introducciones ni texto extra."""

    print("\nIniciando generación de CoT con llama.cpp...")

    # =====================================================================
    # 3. Inferencia
    # =====================================================================
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Generando Razonamiento"):
        if pd.notna(row["reasoning"]) and str(row["reasoning"]).strip() != "":
            continue

        tweet = row["tweet_text"]
        clase_real = row["class_clean"]

        user_prompt = f"### Tweet:\n{tweet}\n\n### Categoría Real:\n{clase_real}\n\nGenera el análisis estructurado."

        try:
            # Uso de la API Chat Completion (similar a OpenAI) integrada en llama-cpp-python
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=200,  # Límite seguro para no desbordar VRAM
                stop=["###", "User:", "<|end_of_turn|>"]  # Stop words de seguridad
            )

            # Extraer el texto generado
            generated_text = response["choices"][0]["message"]["content"].strip()
            df.at[index, "reasoning"] = generated_text

        except Exception as e:
            print(f"\nError en el índice {index}: {e}")
            df.at[index, "reasoning"] = "ERROR"

        # Guardado de seguridad cada 50 iteraciones
        if index % 50 == 0:
            df.to_csv(output_file, index=False, encoding="utf-8")

    # =====================================================================
    # 4. Guardado Final
    # =====================================================================
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n¡Proceso completado! Dataset con CoT guardado en {output_file}")


if __name__ == "__main__":
    main()
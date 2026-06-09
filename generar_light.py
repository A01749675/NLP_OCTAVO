import pandas as pd
from text_cleaner import text_filtering_light


def crear_dataset_ligero():
    # 1. Cargar tus datos CRUDOS / ORIGINALES
    # Cambia "files/data_train_raw.csv" por el nombre exacto de tu archivo original
    ruta_original = "files/data_test_fold1(in).csv"
    ruta_salida = "files/data_test1_light.csv"

    print(f"Cargando datos originales desde {ruta_original}...")
    df = pd.read_csv(ruta_original, encoding="utf-8")

    # 2. Aplicar la limpieza ligera
    print("Aplicando limpieza respetuosa con la sintaxis...")
    # Asumimos que la columna original se llama 'tweet_text'
    df["tweet_text_clean"] = df["tweet_text"].apply(text_filtering_light)

    # 3. Filtrar filas que hayan quedado vacías y guardar
    df = df.dropna(subset=['tweet_text_clean', 'class'])
    df = df[df['tweet_text_clean'].str.strip() != '']

    df.to_csv(ruta_salida, index=False, encoding="utf-8")
    print(f"¡Listo! Nuevo dataset ligero guardado en: {ruta_salida}")


if __name__ == "__main__":
    crear_dataset_ligero()
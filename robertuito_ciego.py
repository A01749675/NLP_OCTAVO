import os
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

# Importaciones de tu propia arquitectura
from paths import resolve_input_path, resolve_output_path, resolve_model_path
from data_loader import get_data
from evaluation import evaluate_model, print_metrics, calculate_auc


def evaluar_prueba_ciega():
    print("\n" + "=" * 60)
    print("INICIANDO EVALUACIÓN CIEGA (BLIND TEST) - ROBERTUITO")
    print("=" * 60)

    # 1. Configuración de Rutas
    ruta_modelo_hf = "./robertuito-embeddings-optimizados"
    archivo_test = "files/data_test1_light.csv"  # Tu archivo de prueba con limpieza ligera
    archivo_train_embeddings = resolve_output_path("data_robertuito_finetuned_embeddings.csv")

    # 2. Recrear el Escalador Matemático (Crucial)
    print("[*] Recuperando el StandardScaler original...")
    X_train, _ = get_data(archivo_train_embeddings)
    scaler = StandardScaler()
    scaler.fit(X_train)

    # 3. Cargar textos de prueba ocultos
    print(f"[*] Cargando datos de prueba desde: {archivo_test}")
    df_test = pd.read_csv(resolve_input_path(archivo_test), encoding="utf-8")

    columna_texto = "tweet_text_clean" if "tweet_text_clean" in df_test.columns else "tweet_text"
    textos_test = df_test[columna_texto].fillna("").tolist()

    y_test = df_test["class"]

    # 4. Extraer Embeddings Nuevos
    print("[*] Pasando textos nuevos por el Transformer optimizado...")
    model_hf = SentenceTransformer(ruta_modelo_hf)
    X_test_raw = model_hf.encode(textos_test, batch_size=32, show_progress_bar=True)

    # 5. Escalar los Embeddings Nuevos
    print("[*] Aplicando escalado a los vectores de prueba...")
    columnas_features = [f"robertuito_feat_{j}" for j in range(X_test_raw.shape[1])]

    X_test_df_raw = pd.DataFrame(X_test_raw, columns=columnas_features)

    # Transformamos usando el escalador original
    X_test_scaled = scaler.transform(X_test_df_raw)

    # Volvemos a empaquetar para los modelos
    X_test = pd.DataFrame(X_test_scaled, columns=columnas_features)

    # 6. Evaluar los Modelos Guardados
    modelos_a_evaluar = ["knn", "rf", "lr"]
    resultados_ciegos = []

    for model_name in modelos_a_evaluar:
        ruta_pkl = resolve_model_path(f"{model_name}-robertuito_finetuned.pkl")

        if not os.path.exists(ruta_pkl):
            print(f"\n[!] ADVERTENCIA: No se encontró el modelo {ruta_pkl}. Saltando...")
            continue

        print("\n" + "-" * 40)
        print(f"EVALUANDO MODELO: {model_name.upper()}")
        print("-" * 40)

        # Cargar el modelo físico (.pkl) previamente entrenado
        artifact = joblib.load(ruta_pkl)
        modelo_entrenado = artifact["model"]

        # Generar predicciones reales a ciegas
        y_pred, results = evaluate_model(model=modelo_entrenado, X_test=X_test, y_test=y_test)
        results["auc"] = calculate_auc(model=modelo_entrenado, X_test=X_test, y_test=y_test)

        resultados_ciegos.append({
            "representation": "robertuito_finetuned",
            "model": model_name,
            "test_type": "blind_fold",
            **results
        })

        print_metrics(results)

    # 7. Guardar el reporte definitivo
    if resultados_ciegos:
        ruta_csv_resultados = resolve_output_path("resultados_CIEGOS_robertuito.csv")
        pd.DataFrame(resultados_ciegos).to_csv(ruta_csv_resultados, index=False, encoding="utf-8")
        print("\n" + "=" * 60)
        print(f"¡Evaluación ciega finalizada!")
        print(f"Reporte definitivo guardado en: {ruta_csv_resultados}")
        print("=" * 60)


if __name__ == "__main__":
    evaluar_prueba_ciega()
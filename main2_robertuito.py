"""
Pipeline Unificado de RoBERTuito.
Este script se encarga de:
1. Generar los embeddings optimizados automáticamente (si no existen).
2. Aplicar el escalado matemático (StandardScaler).
3. Entrenar y evaluar KNN, Random Forest y Regresión Logística.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Importaciones de tu propia arquitectura
from paths import resolve_input_path, resolve_output_path, resolve_model_path
from data_loader import get_data
from evaluation import evaluate_model, print_metrics, calculate_auc

from models.random_forest_model import get_model as get_random_forest
from models.logistic_regression_model import get_model as get_logistic_regression
from models.knn_model import get_model as get_knn


def select_model(model_name, random_state=42):
    """Instancia el modelo clásico solicitado."""
    model_name = model_name.lower()
    if model_name in ["rf", "random_forest"]:
        return get_random_forest(random_state=random_state)
    elif model_name in ["lr", "logistic_regression"]:
        return get_logistic_regression(random_state=random_state)
    elif model_name in ["knn"]:
        return get_knn()
    else:
        raise ValueError("Modelo no válido. Usa 'rf', 'lr' o 'knn'.")


def preparar_embeddings():
    """
    Genera los embeddings usando el modelo optimizado si no se han generado previamente.
    Retorna la ruta del archivo CSV con los embeddings listos para usar.
    """
    output_file_name = "data_robertuito_finetuned_embeddings.csv"
    ruta_salida_final = resolve_output_path(output_file_name)

    # Si ya existen, nos ahorramos el tiempo de procesamiento
    if os.path.exists(ruta_salida_final):
        print(f"[*] Embeddings ya detectados en: {ruta_salida_final}")
        return ruta_salida_final

    print("\n[*] Generando nuevos embeddings desde el modelo optimizado...")
    model_path = "./robertuito-embeddings-optimizados"
    input_file_name = "files/data_train_light.csv"

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ERROR: No se encontró el modelo en '{model_path}'. "
            "Asegúrate de haber ejecutado el script de entrenamiento (Triplet Loss) primero."
        )

    # 1. Cargar el modelo personalizado y los datos
    print("Cargando modelo de SentenceTransformers...")
    model = SentenceTransformer(model_path)

    input_path = resolve_input_path(input_file_name)
    print(f"Leyendo textos limpios desde: {input_path}")
    df = pd.read_csv(input_path, encoding="utf-8")

    columna_texto = "tweet_text_clean" if "tweet_text_clean" in df.columns else "tweet_text"
    textos = df[columna_texto].fillna("").tolist()
    tweet_ids = df["tweet_id"].tolist() if "tweet_id" in df.columns else None
    clases = df["class"].tolist() if "class" in df.columns else None

    # 2. Inferencia y extracción
    print(f"Procesando {len(textos)} tuits en la GPU...")
    matriz_vectores = model.encode(textos, batch_size=32, show_progress_bar=True)

    # 3. Estructuración del CSV compatible
    columnas_features = [f"robertuito_feat_{j}" for j in range(matriz_vectores.shape[1])]
    df_output = pd.DataFrame(matriz_vectores, columns=columnas_features)

    if tweet_ids is not None:
        df_output.insert(0, "tweet_id", tweet_ids)
    if clases is not None:
        df_output["class"] = clases

    df_output.to_csv(ruta_salida_final, index=False, encoding="utf-8")
    print(f"[*] Archivo de características creado exitosamente en: {ruta_salida_final}\n")

    return ruta_salida_final


def run_robertuito_pipeline():
    """Ejecuta los experimentos de ML usando los embeddings preprocesados."""

    # 1. Garantizar que los embeddings existan antes de empezar
    archivo_csv = preparar_embeddings()

    # 2. Definimos exclusivamente los experimentos
    experiments = [
        {"target": "robertuito_finetuned", "model_name": "knn"},
        {"target": "robertuito_finetuned", "model_name": "rf"},
        {"target": "robertuito_finetuned", "model_name": "lr"}
    ]

    all_results = []
    test_size = 0.20
    random_state = 42

    for exp in experiments:
        target = exp["target"]
        model_name = exp["model_name"]

        print("\n" + "=" * 60)
        print(f"Ejecutando experimento: {target.upper()} + {model_name.upper()}")
        print("=" * 60)

        # 3. Cargar datos
        X, y = get_data(archivo_csv)

        # 4. Split del dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # 5. EL ESCALADO (CRUCIAL PARA EMBEDDINGS DENSOS)
        print("Aplicando StandardScaler a los embeddings...")
        columnas = X_train.columns
        scaler = StandardScaler()

        # Ajustamos y transformamos train
        X_train_scaled = scaler.fit_transform(X_train)
        X_train = pd.DataFrame(X_train_scaled, columns=columnas, index=X_train.index)

        # Transformamos test para evitar Data Leakage
        X_test_scaled = scaler.transform(X_test)
        X_test = pd.DataFrame(X_test_scaled, columns=columnas, index=X_test.index)

        # 6. Entrenar el modelo
        print(f"Entrenando modelo {model_name.upper()}...")
        model = select_model(model_name, random_state)
        model.fit(X_train, y_train)

        # 7. Evaluar el modelo
        y_pred, results = evaluate_model(model=model, X_test=X_test, y_test=y_test)
        results["auc"] = calculate_auc(model=model, X_test=X_test, y_test=y_test)

        print_metrics(results)

        # 8. Guardar el modelo físico (.pkl)
        artifact = {
            "model": model,
            "model_name": model_name,
            "target": target,
            "feature_columns": list(columnas)
        }
        ruta_modelo = resolve_model_path(f"{model_name}-{target}.pkl")
        joblib.dump(artifact, ruta_modelo)
        print(f"Modelo guardado en: {ruta_modelo}")

        # 9. Acumular métricas
        results_record = {
            "representation": target,
            "model": model_name,
            **results
        }
        all_results.append(results_record)

    # 10. Guardar el reporte final consolidado
    if all_results:
        df_results = pd.DataFrame(all_results)
        ruta_csv_resultados = resolve_output_path("resultados_robertuito_optimizados.csv")
        df_results.to_csv(ruta_csv_resultados, index=False, encoding="utf-8")
        print("\n" + "=" * 60)
        print(f"¡Todos los experimentos completados exitosamente!")
        print(f"Reporte de métricas guardado en: {ruta_csv_resultados}")
        print("=" * 60)


if __name__ == "__main__":
    run_robertuito_pipeline()
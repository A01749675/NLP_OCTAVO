# NLP_OCTAVO

## Descripción general

`NLP_OCTAVO` es una herramienta en Python para clasificación de tweets en español (ej. etiquetas `control` vs `anorexia`). El proyecto incluye preprocesamiento, extracción de características (TF-IDF, n-grams, Word2Vec, embeddings transformer), pipelines de entrenamiento clásico (RF, LR, KNN), validación y una rama alternativa de clasificación por prompts (LLM/Ollama).

Este README documenta los módulos principales, la arquitectura, cómo ejecutar experimentos y notas de desarrollo.

## Estructura de módulos (resumen)

- `text_cleaner.py` — limpieza y normalización de texto (tokenización, eliminación de stopwords, normalización). Descarga recursos NLTK cuando sea necesario.
- `paths.py` — helpers para resolver rutas de entrada/salida y crear `files/` y `model_files/` de forma segura.
- `vectorizers.py` — funciones para generar representaciones: `tfidf_vectorize`, `hashing_vectorize`, `ngram_vectorize`, `word2vec_vectorize`, combinadores (`all_vectorize`, `tfidf_bigrams_vectorize`, `tfidf_trigrams_vectorize`) y `process_csv()` como enrutador.
- `data_loader.py` — carga CSVs vectorizados y devuelve `(X, y)` listos para entrenar.
- `models/` — contenedor de constructores de modelos: `random_forest_model`, `logistic_regression_model`, `knn_model` (cada uno expone `get_model()` o `get_model(random_state=...)`).
- `main.py` — orquestador principal: `train_and_plot()`, `test_knn_model()`, `run_experiments()` y `save_model()`.
- `evaluation.py` — métricas (accuracy, precision/recall macro, especificidad), funciones gráficas (distribución de clases, matriz de confusión, curvas ROC, importancias/coeficientes) y utilidades de visualización.
- `model_validation.py` — carga artefactos guardados (`.pkl`), alinea columnas de features, y valida modelos sobre conjuntos de test.
- `ai_classifier.py` — flujo alternativo basado en prompts con Ollama (lectura de tweets, clasificación few-shot / chain-of-thought, cálculo de matriz de confusión para salidas del LLM).
- `train_llm_cot.py` — fine-tuning de Llama 3.2 con Chain-of-Thought (CoT) y QLoRA para generar razonamientos intermedios antes de la clasificación final.
- `cot_gen.py` — utilidades y generación de prompts/datos de Chain-of-Thought para el flujo LLM.
- `main_bert.py` / `main_robertuito.py` — scripts para ajustar (fine-tune) BETO y RoBERTuito respectivamente.
- `tests/` — suite `unittest` con pruebas unitarias para `paths`, `vectorizers`, `evaluation`, `main`, etc.

## Flujo de ejecución (alto nivel)

1. Limpieza: `text_cleaner.py` → genera `files/data_train_cleaned.csv` (y variantes).
2. Vectorización: `vectorizers.process_csv(input_file, target)` → genera archivos en `files/` (por ejemplo `data_train_tfidf.csv`, `data_train_hashing.npz`, `data_train_word2vec.csv`).
3. Entrenamiento: `main.train_and_plot()` selecciona modelo (`select_model()`), entrena, guarda artefacto (`model_files/`), evalúa (`evaluation.py`) y genera gráficos.
4. Validación: `model_validation.py` compara artefactos guardados contra datos de validación/producción.
5. Alternativa LLM:
   - `ai_classifier.py` para clasificación por prompts con Ollama.
   - `train_llm_cot.py` para fine-tuning QLoRA de Llama 3.2 con Chain-of-Thought.
   - `cot_gen.py` para utilidades de generación de prompts/datos CoT.

## Representaciones soportadas

Las representaciones que puede producir el pipeline son:

- `tfidf`, `ngrams`, `bigrams`, `trigrams`
- `word2vec`, `all` (combinación TF-IDF + n-grams + Word2Vec)
- `tfidf_bigrams`, `tfidf_trigrams`
- `beto`, `beto_finetuned`, `robertuito`, `robertuito_finetuned`

Usa `process_csv(input_file, target)` para generar la representación deseada.

## Uso rápido

Instalar dependencias:

```bash
pip install -r requirements.txt
```

Limpiar datos (ejemplo):

```bash
python text_cleaner.py
```

Vectorizar y entrenar un modelo RF (ejemplo):

```python
from main import train_and_plot

train_and_plot(
    input_file="files/data_train_cleaned.csv",
    target="tfidf",
    model_name="rf",
    random_state=42
)
```

Ejecutar la suite de experimentos:

```bash
python main.py
```

Ejecutar la barrida KNN:

```python
from main import test_knn_model

test_knn_model(input_file="files/data_train_cleaned.csv")
```

Validar artefactos guardados:

```bash
python model_validation.py
```

Clasificación por prompts (requiere Ollama):

```python
from ai_classifier import read_tweets, classify_tweet, classify_tweet_few_shot, classify_chain_of_thought, calculate_confusion_matrix

# Ejemplo: leer y clasificar
```

Fine-tuning de transformadores y CoT (puede requerir GPU):

```bash
python main_bert.py
python main_robertuito.py
python train_llm_cot.py
```

## Desarrollo y testing

- Tests: `python -m unittest discover -s tests -p 'test_*.py'`
- Formato/linters: aplicar tu estándar (black/flake8) según prefieras.
- Asegúrate de tener NLTK instalado y sus recursos (ej. `punkt`, `stopwords`) descargados; los scripts los solicitan automáticamente si faltan.

## Convenciones de rutas y archivos

- Entrada por defecto: `files/data_train_cleaned.csv` (y `files/data_train_cleaned2.csv` para algunos flujos).
- Salidas vectorizadas: guardadas en `files/`.
- Artefactos de modelos clásicos: `model_files/{model_name}-{target}.pkl`.
- Modelos fine-tuned: `modelo_beto_final/`, `modelo_robertuito_final/`.
- Resultados de experimentos: `files/all_experiments.csv`, `files/knn_performance.csv`, `files/model_evaluation_results.csv`.

## Notas operativas

- `ai_classifier.py` requiere sitio local de Ollama si se usa ese backend.
- `train_llm_cot.py` entrena un modelo Llama 3.2 en 4-bit con QLoRA y Chain-of-Thought. Necesita datos `files/data_train_cot.csv` con razonamiento previo (`reasoning`) y puede requerir GPU/memoria adicional.
- Los scripts de fine-tune usan `transformers` y pueden requerir memoria/GPU; el código intenta detectar CUDA y caerá a CPU si no hay GPU disponible.
- `vectorizers.word2vec_vectorize()` buscará `model_files/WORD2VEC.model`; si no existe, entrenará un Word2Vec local y lo guardará.

## FAQ rápidas

- ¿Cómo fuerzo TF-IDF en lugar de Hashing? — `process_csv(..., target='tfidf', use_hashing=False)`.
- ¿Dónde están las columnas de features? — `save_model` guarda el `artifact['feature_columns']` dentro del `.pkl`.

## Contribuciones

Si deseas contribuir: forkea, añade tests para nuevas funciones y abre un PR con la descripción del cambio y pruebas que verifiquen el comportamiento.

## Licencia

Agrega un archivo `LICENSE` si planeas publicar el repositorio. Actualmente no hay licencia explícita en este repositorio.

---

Si quieres, puedo:

- Añadir una sección de ejemplos completos (notebook o `examples/`) para reproducir experimentos pequeños.
- Generar un `CONTRIBUTING.md` y `CODE_OF_CONDUCT.md` básicos.
- Añadir instrucciones de docker/entorno (requirements/Makefile) para reproducción más sencilla.

¿Qué deseas que añada a continuación?

"""Feature generation utilities for text classification.

This module builds the vectorized representations used by the models,
including TF-IDF, count-based n-grams, Word2Vec embeddings, and combined
feature sets. It also exposes ``process_csv`` as the entry point for turning
cleaned CSV files into model-ready feature files.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.decomposition import PCA
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
import os

from paths import resolve_input_path, resolve_output_path, resolve_model_path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# TF-IDF VECTORIZER
# ---------------------------------------------------------
def tfidf_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_tfidf.csv",
    ngram_range=(1, 1)
):
    """
    Generates a TF-IDF representation from cleaned tweet texts.

    Parameters
    ----------
    texts : list
        List of cleaned tweet texts.

    tweet_ids : list
        List of tweet IDs.

    classes : list, optional
        List of class labels.

    output_file : str
        Name of the output CSV file.

    ngram_range : tuple
        Range of n-grams to use in TF-IDF.

    Returns
    -------
    pd.DataFrame
        DataFrame containing TF-IDF features and metadata.
    """

    output_file = resolve_output_path(output_file)

    tfidf = TfidfVectorizer(ngram_range=ngram_range)

    tfidf_matrix = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    tfidf_matrix = tfidf_matrix.astype(np.float32)
    tfidf_df = pd.DataFrame.sparse.from_spmatrix(
        tfidf_matrix,
        columns=[f"tfidf_{name}" for name in feature_names]
    )

    tfidf_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        tfidf_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        tfidf_df.insert(0, "class", classes)

    tfidf_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"TF-IDF data saved to {output_file}")
    print(f"Number of TF-IDF features: {len(feature_names)}")

    return tfidf_df


# ---------------------------------------------------------
# N-GRAM VECTORIZER
# ---------------------------------------------------------
def ngram_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_ngrams.csv",
    ngram_range=(1, 3)
):
    """
    Generates a count-based n-gram representation from cleaned tweets.

    Parameters
    ----------
    texts : list
        List of cleaned tweet texts.

    tweet_ids : list
        List of tweet IDs.

    classes : list, optional
        List of class labels.

    output_file : str
        Name of the output CSV file.

    ngram_range : tuple
        Range of n-grams to use.

    Returns
    -------
    pd.DataFrame
        DataFrame containing n-gram features and metadata.
    """

    output_file = resolve_output_path(output_file)

    vectorizer = CountVectorizer(ngram_range=ngram_range)

    ngram_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    ngram_matrix = ngram_matrix.astype(np.uint16)
    ngram_df = pd.DataFrame.sparse.from_spmatrix(
        ngram_matrix,
        columns=[f"ngram_{name}" for name in feature_names]
    )

    ngram_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        ngram_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        ngram_df.insert(0, "class", classes)

    ngram_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"N-gram data saved to {output_file}")
    print(f"Number of n-gram features: {len(feature_names)}")

    return ngram_df


# ---------------------------------------------------------
# WORD2VEC VECTORIZER
# ---------------------------------------------------------
def word2vec_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_word2vec.csv",
    vector_size=50,
    window=10,
    min_count=1,
    epochs=100
):
    """
    Generates one Word2Vec vector per tweet.

    This function trains a local Skip-gram Word2Vec model using:
    1. The cleaned tweets from the dataset.
    2. An additional domain-specific anorexia-related document.

    Each tweet is represented by averaging the vectors of its tokens.

    Parameters
    ----------
    texts : list
        List of cleaned tweet texts.

    tweet_ids : list
        List of tweet IDs.

    classes : list, optional
        List of class labels.

    output_file : str
        Name of the output CSV file.

    vector_size : int
        Number of dimensions for the Word2Vec embeddings.

    window : int
        Context window size.

    min_count : int
        Minimum frequency required for a word to be included.

    epochs : int
        Number of training epochs.

    Returns
    -------
    pd.DataFrame
        DataFrame containing Word2Vec features and metadata.
    """

    output_file = resolve_output_path(output_file)
    
    model_file = resolve_model_path("WORD2VEC.model")
    
    # 1. Prepare Tokenization helper
    def clean_and_tokenize(text):
        if not text or pd.isna(text): return []
        text = str(text).lower()
        # Remove non-alphanumeric but keep spaces
        text = "".join(c for c in text if c.isalnum() or c.isspace())
        return word_tokenize(text)

    # 2. Tokenize the input tweets
    tweet_tokens = [clean_and_tokenize(text) for text in texts]

    # 3. Check if model exists or needs training
    if os.path.exists(model_file):
        print(f"Loading existing Word2Vec model from {model_file}...")
        model = Word2Vec.load(model_file)
    else:
        print("Training new Word2Vec model with expanded context...")
        # Load the base domain document (tweets context)
        word2vec_text_path = resolve_input_path("word2vecText.txt")
        try:
            with open(word2vec_text_path, "r", encoding="utf-8") as f:
                domain_document = f.read()
        except FileNotFoundError:
            domain_document = ""
            print(f"Warning: {word2vec_text_path} not found. Using only tweets for training.")

        # ADDING THE NEW CONTEXT (The Monologue)
        # You can also save this to a file and read it, but adding it here 
        # ensures the model learns these specific semantic relationships.
        new_monologue = """
        Hoy desperté sintiendo que mi mente empezó antes que mi cuerpo. 
        Es extraño vivir así, como si mi cabeza estuviera siempre ocupada 
        por la comida, el peso, el espejo, la ropa, la imagen y la culpa.
        
        """
        
        domain_tokens = clean_and_tokenize(domain_document)
        monologue_tokens = clean_and_tokenize(new_monologue)

        # Build training corpus: list of lists
        # We include tweet tokens, the domain doc as a "sentence", 
        # and the monologue as a "sentence".
        training_corpus = tweet_tokens + [domain_tokens] + [monologue_tokens]
        
        # Train the model
        model = Word2Vec(
            sentences=training_corpus,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            sg=1, # Skip-gram is usually better for small datasets with deep context
            workers=4,
            epochs=epochs
        )
        
        model.save(model_file)
        print("Model trained and saved.")

    # 4. Vectorize tweets (Inference)
    def tweet_to_vector(tokens):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if not vectors:
            return np.zeros(vector_size)
        return np.mean(vectors, axis=0)

    tweet_vectors = [tweet_to_vector(tokens) for tokens in tweet_tokens]

    # 5. Create DataFrame and Save
    feature_names = [f"word2vec_{i}" for i in range(vector_size)]
    word2vec_df = pd.DataFrame(tweet_vectors, columns=feature_names)
    
    word2vec_df.insert(0, "tweet_text_clean", texts)
    if tweet_ids is not None: word2vec_df.insert(0, "tweet_id", tweet_ids)
    if classes is not None: word2vec_df.insert(0, "class", classes)

    word2vec_df.to_csv(output_file, index=False, encoding="utf-8")
    return word2vec_df


def _combine_tfidf_and_ngrams(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_tfidf_ngrams.csv",
    tfidf_ngram_range=(1, 1),
    count_ngram_range=(2, 2),
    label="Combined TF-IDF + N-gram"
):
    """Shared helper for TF-IDF + count-based n-gram combinations."""

    output_file = resolve_output_path(output_file)

    tfidf_df = tfidf_vectorize(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file="temporary_tfidf.csv",
        ngram_range=tfidf_ngram_range
    )

    ngram_df = ngram_vectorize(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file="temporary_ngrams.csv",
        ngram_range=count_ngram_range
    )

    combined_df = pd.concat(
        [
            tfidf_df.drop(columns=["tweet_text_clean", "tweet_id", "class"], errors="ignore"),
            ngram_df.drop(columns=["tweet_text_clean", "tweet_id", "class"], errors="ignore")
        ],
        axis=1
    )

    combined_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        combined_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        combined_df.insert(0, "class", classes)

    combined_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"{label} data saved to {output_file}")
    print(f"Number of TF-IDF features: {tfidf_df.shape[1] - 3}")
    print(f"Number of N-gram features: {ngram_df.shape[1] - 3}")
    # print(f"Total features: {combined_df.shape[1] - 3}")

    return combined_df


def tfidf_bigrams_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_tfidf_bigrams.csv",
    tfidf_ngram_range=(2, 2),
    count_ngram_range=(2, 2)
):
    """Combines TF-IDF features with bigrams."""

    return _combine_tfidf_and_ngrams(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file=output_file,
        tfidf_ngram_range=tfidf_ngram_range,
        count_ngram_range=count_ngram_range,
        label="Combined TF-IDF + Bigrams"
    )


def tfidf_trigrams_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_tfidf_trigrams.csv",
    tfidf_ngram_range=(1, 1),
    count_ngram_range=(3, 3)
):
    """Combines TF-IDF features with trigrams."""

    return _combine_tfidf_and_ngrams(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file=output_file,
        tfidf_ngram_range=tfidf_ngram_range,
        count_ngram_range=count_ngram_range,
        label="Combined TF-IDF + Trigrams"
    )


def tfidf_ngram_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_tfidf_ngrams.csv",
    tfidf_ngram_range=(1, 1),
    count_ngram_range=(3, 3)
):
    """Backward-compatible alias for trigram-based TF-IDF + n-gram output."""

    return tfidf_trigrams_vectorize(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file=output_file,
        tfidf_ngram_range=tfidf_ngram_range,
        count_ngram_range=count_ngram_range
    )


def all_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_train_all.csv",
    tfidf_ngram_range=(1, 1),
    count_ngram_range=(3, 3)
):
    """
    Combines TF-IDF, n-grams, and Word2Vec into one representation.
    
        args:
            texts (list): List of cleaned tweet texts.
            tweet_ids (list): List of tweet IDs.
            classes (list, optional): List of class labels.
            output_file (str): Name of the output CSV file.
            tfidf_ngram_range (tuple): Range of n-grams
        returns:
            pd.DataFrame: A DataFrame containing the combined TF-IDF, n-gram, and Word2Vec features along with metadata.
    """

    output_file = resolve_output_path(output_file)

    # -----------------------------
    # 1. TF-IDF
    # -----------------------------
    tfidf = TfidfVectorizer(ngram_range=tfidf_ngram_range)
    tfidf_matrix = tfidf.fit_transform(texts)
    tfidf_features = tfidf.get_feature_names_out()

    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{name}" for name in tfidf_features]
    )

    # -----------------------------
    # 2. N-grams
    # -----------------------------
    count = CountVectorizer(ngram_range=count_ngram_range)
    count_matrix = count.fit_transform(texts)
    count_features = count.get_feature_names_out()

    ngram_df = pd.DataFrame(
        count_matrix.toarray(),
        columns=[f"ngram_{name}" for name in count_features]
    )

    # -----------------------------
    # 3. Word2Vec
    # -----------------------------
    word2vec_df = word2vec_vectorize(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file="temporary_word2vec.csv"
    )

    columns_to_drop = [
        "class",
        "tweet_id",
        "tweet_text_clean"
    ]

    word2vec_features_df = word2vec_df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    # -----------------------------
    # 4. Combine
    # -----------------------------
    all_df = pd.concat(
        [
            tfidf_df.reset_index(drop=True),
            ngram_df.reset_index(drop=True),
            word2vec_features_df.reset_index(drop=True)
        ],
        axis=1
    )

    all_df.insert(0, "tweet_text_clean", texts)

    if tweet_ids is not None:
        all_df.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        all_df.insert(0, "class", classes)

    all_df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Combined TF-IDF + N-gram + Word2Vec data saved to {output_file}")
    print(f"Number of TF-IDF features: {len(tfidf_features)}")
    print(f"Number of N-gram features: {len(count_features)}")
    print(f"Number of Word2Vec features: {word2vec_features_df.shape[1]}")
    print(f"Total features: {all_df.shape[1] - 3}")

    return all_df


def _run_beto_vectorization(
    texts,
    tweet_ids,
    classes,
    output_file,
    ruta_modelo,
    batch_size,
    require_local
):
    """Internal helper to extract optimized BERT embeddings from a Spanish BETO model."""

    if ruta_modelo and os.path.exists(ruta_modelo):
        model_source = ruta_modelo
        print(f"Loading BETO from local model: {ruta_modelo}")
    elif require_local:
        raise FileNotFoundError(
            f"Fine-tuned BETO model directory not found: {ruta_modelo}"
        )
    else:
        # NOTE: If your data preprocessing fully lowercases the text, 
        # consider changing this to "dccuchile/bert-base-spanish-wwm-uncased"
        model_source = "dccuchile/bert-base-spanish-wwm-cased"
        print(
            f"Local model not found at {ruta_modelo}. Falling back to base BETO: {model_source}"
        )

    tokenizer = BertTokenizer.from_pretrained(model_source)
    
    # IMPROVEMENT 1: Enable output_hidden_states to capture deep semantic features
    modelo = BertModel.from_pretrained(model_source, output_hidden_states=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Procesando con: {device}")

    modelo.to(device)
    modelo.eval()

    todos_los_vectores = []
    print("Extrayendo vectores optimizados (Last 4 Layers + Mean Pooling)...")

    for i in tqdm(range(0, len(texts), batch_size)):
        lote_textos = texts[i : i + batch_size]
        inputs = tokenizer(
            lote_textos,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = modelo(**inputs)

        # IMPROVEMENT 2: Extract and average the last 4 hidden layers
        # outputs.hidden_states is a tuple containing the initial embeddings + 12 layers
        hidden_states = outputs.hidden_states
        last_four_layers = torch.stack(hidden_states[-4:], dim=0)  # Shape: [4, batch_size, seq_len, 768]
        layer_averaged = torch.mean(last_four_layers, dim=0)       # Shape: [batch_size, seq_len, 768]

        # IMPROVEMENT 3: Perform Mean Pooling while ignoring padding tokens
        attention_mask = inputs["attention_mask"].unsqueeze(-1)    # Shape: [batch_size, seq_len, 1]
        mask_expanded = attention_mask.expand(layer_averaged.size()).float()
        
        # Zero out padding token representations
        masked_embeddings = layer_averaged * mask_expanded
        
        # Sum along the sequence length (axis 1)
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        
        # Count actual valid tokens, clamp to prevent division by zero on empty inputs
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        
        # Calculate final mean vectors
        lote_vectores = (sum_embeddings / sum_mask).cpu().numpy()
        todos_los_vectores.append(lote_vectores)

    matriz_vectores = np.vstack(todos_los_vectores)

    print("Armando el dataset final...")
    columnas_features = [f"beto_feat_{j}" for j in range(matriz_vectores.shape[1])]
    df_output = pd.DataFrame(matriz_vectores, columns=columnas_features)
    df_output.insert(0, "tweet_id", tweet_ids)

    if classes is not None:
        df_output["class"] = classes

    output_file = resolve_output_path(output_file)
    df_output.to_csv(output_file, index=False, encoding="utf-8")

    print(f"¡Listo! Archivo guardado exitosamente en: {output_file}")
    print(f"Forma del dataset: {df_output.shape} (Filas, Columnas)")

    return df_output


def beto_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_beto_embeddings.csv",
    ruta_modelo=None,
    batch_size=32
):
    """Generate BETO embeddings using a local model if available, else fallback to base BETO."""
    return _run_beto_vectorization(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file=output_file,
        ruta_modelo=ruta_modelo,
        batch_size=batch_size,
        require_local=False,
    )


def beto_finetuned_vectorize(
    texts,
    tweet_ids,
    classes=None,
    output_file="data_beto_finetuned_embeddings.csv",
    ruta_modelo="./modelo_beto_final",
    batch_size=32
):
    """Generate BETO embeddings using the fine-tuned local BETO model.

    This function will raise a FileNotFoundError if the specified fine-tuned
    model directory does not exist.
    """
    return _run_beto_vectorization(
        texts=texts,
        tweet_ids=tweet_ids,
        classes=classes,
        output_file=output_file,
        ruta_modelo=ruta_modelo,
        batch_size=batch_size,
        require_local=True,
    )

# =====================================================================
# NUEVO: ENRUTAMIENTO Y VECTORIZADORES DE ROBERTUITO (ROBERTA)
# =====================================================================
def _run_robertuito_vectorization(texts, tweet_ids, classes, output_file, ruta_modelo, batch_size, require_local):
    """Extractor de embeddings optimizado usando Mean Pooling para RoBERTuito."""
    if ruta_modelo and os.path.exists(ruta_modelo):
        model_source = ruta_modelo
        print(f"Cargando RoBERTuito local/finedtuned desde: {ruta_modelo}")
    elif require_local:
        raise FileNotFoundError(f"No se encontró el directorio del modelo ajustado: {ruta_modelo}")
    else:
        model_source = "pysentimiento/robertuito-base-cased"
        print(f"Cargando RoBERTuito Base desde Hugging Face: {model_source}")

    # RoBERTuito requiere Auto classes obligatoriamente en vez de Bert classes
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    modelo = AutoModel.from_pretrained(model_source)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(device)
    modelo.eval()

    todos_los_vectores = []
    print("Extrayendo embeddings estáticos con Mean Pooling...")

    for i in tqdm(range(0, len(texts), batch_size)):
        lote_textos = texts[i : i + batch_size]
        inputs = tokenizer(
            lote_textos, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = modelo(**inputs)
        
        # Implementación de Mean Pooling (Ignora tokens de padding para evitar ruido geométrico)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)
        mask_expanded = attention_mask.expand(token_embeddings.size()).float()
        
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        
        lote_vectores = (sum_embeddings / sum_mask).cpu().numpy()
        todos_los_vectores.append(lote_vectores)

    matriz_vectores = np.vstack(todos_los_vectores)

    print("Construyendo dataframe estructurado para la tubería...")
    columnas_features = [f"robertuito_feat_{j}" for j in range(matriz_vectores.shape[1])]
    df_output = pd.DataFrame(matriz_vectores, columns=columnas_features)
    
    if tweet_ids is not None:
        df_output.insert(0, "tweet_id", tweet_ids)
    if classes is not None:
        df_output["class"] = classes

    output_file = resolve_output_path(output_file)
    df_output.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Dataset guardado en: {output_file} | Forma: {df_output.shape}")
    
    return df_output

def robertuito_vectorize(texts, tweet_ids, classes=None, output_file="data_robertuito_embeddings.csv", ruta_modelo=None, batch_size=32):
    return _run_robertuito_vectorization(texts, tweet_ids, classes, output_file, ruta_modelo, batch_size, require_local=False)

def robertuito_finetuned_vectorize(texts, tweet_ids, classes=None, output_file="data_robertuito_finetuned_embeddings.csv", ruta_modelo="./modelo_robertuito_final", batch_size=32):
    return _run_robertuito_vectorization(texts, tweet_ids, classes, output_file, ruta_modelo, batch_size, require_local=True)





# ---------------------------------------------------------
# PROCESS CSV
# ---------------------------------------------------------
def process_csv(input_file, target, only_train=False, test_size=0.2, random_state=42):
    """
    Reads a cleaned CSV file and generates the selected vectorized file.

    Available targets:
    - "tfidf"
    - "ngrams"
    - "word2vec"
    - "all"
    - "beto"

    Parameters
    ----------
    input_file : str
        CSV file containing tweet_text_clean and class.

    target : str
        Type of vectorization to apply.

    Returns
    -------
    str
        Name of the generated vectorized CSV file.
    """

    input_path = resolve_input_path(input_file)
    df = pd.read_csv(input_path, encoding="utf-8")

    if "tweet_text_clean" not in df.columns:
        raise ValueError("The input file must contain a 'tweet_text_clean' column.")

    if "class" not in df.columns:
        raise ValueError("The input file must contain a 'class' column.")

    texts = df["tweet_text_clean"].fillna("").tolist()
    tweet_ids = df["tweet_id"].tolist() if "tweet_id" in df.columns else None
    classes = df["class"].tolist()

    # If requested, return only the training portion (to avoid embedding test rows)
    if only_train:
        stratify = classes if classes is not None else None
        X_train_texts, _, y_train, _, train_ids, _ = train_test_split(
            texts,
            classes,
            tweet_ids,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

        texts = X_train_texts
        classes = y_train
        tweet_ids = train_ids

    match target:
        case "tfidf":
            file_name = resolve_output_path("data_train_tfidf.csv")

            tfidf_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                ngram_range=(1, 1)
            )

        case "ngrams" | "trigrams":
            file_name = resolve_output_path(
                "data_train_ngrams.csv" if target == "ngrams" else "data_train_trigrams.csv"
            )

            ngram_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                ngram_range=(3, 3)
            )

        case "bigrams":
            file_name = resolve_output_path("data_train_bigrams.csv")

            ngram_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                ngram_range=(2, 2)
            )

        case "word2vec":
            file_name = resolve_output_path("data_train_word2vec.csv")

            word2vec_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                vector_size=50,
                window=10,
                min_count=1,
                epochs=100
            )

        case "all":
            file_name = resolve_output_path("data_train_all.csv")

            all_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                tfidf_ngram_range=(1, 1),
                count_ngram_range=(3, 3)
            )
        case "tfidf_bigrams":
            file_name = resolve_output_path("data_train_tfidf_bigrams.csv")

            tfidf_bigrams_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                tfidf_ngram_range=(1, 2),
                count_ngram_range=(2, 2)
            )

        case "tfidf_trigrams":
            file_name = resolve_output_path("data_train_tfidf_trigrams.csv")

            tfidf_trigrams_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                tfidf_ngram_range=(1, 3),
                count_ngram_range=(3, 3)
            )
        case "beto":
            file_name = resolve_output_path("data_beto_embeddings.csv")
            beto_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                batch_size=32
            )
        case "beto_finetuned":
            file_name = resolve_output_path("data_beto_finetuned_embeddings.csv")
            beto_finetuned_vectorize(
                texts=texts,
                tweet_ids=tweet_ids,
                classes=classes,
                output_file=file_name,
                ruta_modelo="./modelo_beto_final",
                batch_size=32
            )
        # --- NUEVOS CASOS INTEGRADOS ---
        case "robertuito":
            file_name = resolve_output_path("data_robertuito_embeddings.csv")
            robertuito_vectorize(texts=texts, tweet_ids=tweet_ids, classes=classes, output_file=file_name)
        case "robertuito_finetuned":
            file_name = resolve_output_path("data_robertuito_finetuned_embeddings.csv")
            robertuito_finetuned_vectorize(texts=texts, tweet_ids=tweet_ids, classes=classes, output_file=file_name, ruta_modelo="./modelo_robertuito_final")
        case _:
            raise ValueError(
                "Invalid target. Use 'tfidf', 'ngrams', 'bigrams', 'trigrams', 'tfidf_bigrams', 'tfidf_trigrams', 'word2vec', 'beto', 'beto_finetuned', or 'all'."
            )

    return file_name


# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    generated_file = process_csv(
        input_file="files/data_train_cleaned.csv",
        target="word2vec"
    )

    print(f"Generated file: {generated_file}")
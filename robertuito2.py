"""Fine-tuning of SentenceTransformer embeddings using Triplet Loss.

This script builds a custom SentenceTransformer model from a base
RoBERTuito model, processes a training dataset of tweets, and
optimizes the latent space using Batch Hard Triplet Loss to better
separate the target classes.
"""

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader, Dataset


class TuitsDataset(Dataset):
    """PyTorch Dataset wrapper for sentence transformer training examples.

    Parameters
    ----------
    ejemplos : list
        A list of InputExample objects containing the text and corresponding label.
    """

    def __init__(self, ejemplos):
        self.ejemplos = ejemplos

    def __len__(self):
        return len(self.ejemplos)

    def __getitem__(self, idx):
        return self.ejemplos[idx]


def entrenar_embeddings():
    """Train and save a customized SentenceTransformer model.

    This function loads a base RoBERTuito model, restricts its sequence length,
    loads and cleans the training dataset, maps textual labels to integers,
    and fine-tunes the embeddings using Batch Hard Triplet Loss. The resulting
    model is saved to a local directory.

    Returns
    -------
    None
    """
    # 1. Construir el modelo por módulos fijando el límite de forma estricta
    modelo_base = "pysentimiento/robertuito-base-uncased"
    word_embedding_model = models.Transformer(modelo_base, max_seq_length=128)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    # Ensamblamos el modelo final uniendo el transformer con la capa de pooling
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2. Cargar tus datos limpios
    df = pd.read_csv("files/data_train_light.csv")

    # Eliminar filas con valores nulos en columnas críticas para el entrenamiento
    df = df.dropna(subset=['tweet_text_clean', 'class'])

    label_map = {
        "control": 0,
        "anorexia": 1
    }

    train_examples = []

    # Transformar las filas del DataFrame en objetos InputExample requeridos por SentenceTransformers
    for index, row in df.iterrows():
        texto = str(row['tweet_text_clean'])

        # Omitir textos vacíos o excesivamente cortos
        if len(texto.strip()) < 2:
            continue

        etiqueta_texto = str(row['class']).strip().lower()

        # Filtrar y mapear únicamente las etiquetas válidas definidas en label_map
        if etiqueta_texto in label_map:
            etiqueta_numerica = label_map[etiqueta_texto]
            train_examples.append(InputExample(texts=[texto], label=etiqueta_numerica))

    # 3. Crear el DataLoader
    train_dataset = TuitsDataset(train_examples)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    # 4. Configurar la función de pérdida
    # BatchHardTripletLoss optimiza el espacio latente acercando ejemplos de la misma clase
    # y alejando los de clases distintas dentro de cada batch
    train_loss = losses.BatchHardTripletLoss(model=model)

    # 5. Entrenar el modelo
    print("Iniciando optimización del espacio latente...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        show_progress_bar=True
    )

    # 6. Guardar el modelo
    output_path = "./robertuito-embeddings-optimizados"
    model.save(output_path)
    print(f"Modelo guardado en {output_path}")


if __name__ == "__main__":
    entrenar_embeddings()
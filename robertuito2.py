import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader, Dataset


class TuitsDataset(Dataset):
    def __init__(self, ejemplos):
        self.ejemplos = ejemplos

    def __len__(self):
        return len(self.ejemplos)

    def __getitem__(self, idx):
        return self.ejemplos[idx]


def entrenar_embeddings():
    # 1. Construir el modelo por módulos fijando el límite de forma estricta
    modelo_base = "pysentimiento/robertuito-base-uncased"
    word_embedding_model = models.Transformer(modelo_base, max_seq_length=128)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

    # Ensamblamos el modelo final
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # 2. Cargar tus datos limpios
    df = pd.read_csv("files/data_train_light.csv")

    df = df.dropna(subset=['tweet_text_clean', 'class'])

    label_map = {
        "control": 0,
        "anorexia": 1
    }

    train_examples = []
    for index, row in df.iterrows():
        texto = str(row['tweet_text_clean'])
        if len(texto.strip()) < 2:
            continue

        etiqueta_texto = str(row['class']).strip().lower()

        if etiqueta_texto in label_map:
            etiqueta_numerica = label_map[etiqueta_texto]
            train_examples.append(InputExample(texts=[texto], label=etiqueta_numerica))

    # 3. Crear el DataLoader
    train_dataset = TuitsDataset(train_examples)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)

    # 4. Configurar la función de pérdida
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
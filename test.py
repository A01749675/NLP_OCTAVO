"""Comparison of base and fine-tuned BERT embeddings.

This script loads a base Spanish BERT model (BETO) and a locally
fine-tuned version of it. It processes a sample text through both
models to extract their [CLS] token embeddings and computes the
maximum and mean absolute differences between the resulting vectors
to evaluate how much the latent space shifted after fine-tuning.
"""

import torch
from transformers import BertTokenizer, BertModel

# Definir el texto de prueba para la extracción de embeddings
sample = ["ejemplo de texto para prueba"]

# Cargar el tokenizador y el modelo base preentrenado (BETO)
tok = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model_base = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Tokenizar el texto de entrada generando los tensores de PyTorch requeridos
inp = tok(sample, return_tensors="pt", padding=True, truncation=True)

# Extraer el embedding del token [CLS] (posición 0 de la secuencia) del modelo base y convertirlo a NumPy
o_base = model_base(**inp).last_hidden_state[:,0,:].detach().numpy()

# Cargar el tokenizador y el modelo ajustado (fine-tuned) desde el directorio local
tok2 = BertTokenizer.from_pretrained("./modelo_beto_final")
model_ft = BertModel.from_pretrained("./modelo_beto_final")

# Procesar los mismos inputs tokenizados por el modelo ajustado y extraer su embedding [CLS]
o_ft = model_ft(**inp).last_hidden_state[:,0,:].detach().numpy()

# Calcular e imprimir las diferencias absolutas máxima y promedio entre ambos vectores latentes
print("max abs diff:", (o_base - o_ft).max(), "mean abs diff:", (o_base - o_ft).mean())
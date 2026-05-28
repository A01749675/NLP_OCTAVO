import torch
from transformers import BertTokenizer, BertModel

sample = ["ejemplo de texto para prueba"]

# base
tok = BertTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model_base = BertModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
inp = tok(sample, return_tensors="pt", padding=True, truncation=True)
o_base = model_base(**inp).last_hidden_state[:,0,:].detach().numpy()

# finetuned (local)
tok2 = BertTokenizer.from_pretrained("./modelo_beto_final")
model_ft = BertModel.from_pretrained("./modelo_beto_final")
o_ft = model_ft(**inp).last_hidden_state[:,0,:].detach().numpy()

print("max abs diff:", (o_base - o_ft).max(), "mean abs diff:", (o_base - o_ft).mean())
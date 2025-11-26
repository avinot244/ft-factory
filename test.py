import torch
import json
from tqdm import tqdm
import torch.nn.functional as F
from services.huggingface.contrastive.model_loader import PredictionHead
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

model : PredictionHead = PredictionHead(input_dim=2048, output_dim=2048)
model.load_state_dict(torch.load("output/v11/model_epoch_3_step_1201_0.2539.pth", map_location=torch.device('cpu')))
model = model.eval()


champion_rationales = []

# Ensure we have a list of entries
path = "./data/champion_rationales.jsonl"
champion_rationales = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        champion_rationales.append(json.loads(line))    

names = []
rationales = []
for entry in champion_rationales:
    names.append(entry["name"])
    rationales.append(entry["rationale"])

# Compute embeddings (batching would be recommended for large lists)
embeddings = []
for t in tqdm(rationales[:1], desc="Embedding rationales"):
    print(t)
    with torch.no_grad():
        emb = model(t)
    emb = emb.detach().cpu().to(torch.float32).squeeze()
    # ensure 1D numpy vector
    # emb = emb.numpy().reshape(-1)
    embeddings.append(emb)

print(embeddings[0].shape)
# if not embeddings:
#     raise ValueError("No embeddings computed; check champion_rationales contents.")

# emb_matrix = np.vstack(embeddings)  # shape (N, D)
# sim_matrix = cosine_similarity(emb_matrix)  # shape (N, N)

# # Plot heatmap
# plt.figure(figsize=(max(8, len(names) * 0.25), max(6, len(names) * 0.25)))
# sns.heatmap(sim_matrix, xticklabels=names, yticklabels=names, cmap="flare", vmin=-1, vmax=1)
# plt.xticks(rotation=90)
# plt.yticks(rotation=0)
# plt.title("Cosine similarity between champion rationales")
# plt.tight_layout()
# plt.savefig("champion_rationales_similarity_heatmap.png", dpi=200)
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
model.load_state_dict(torch.load("output/v13/model_epoch_5_step_3001_0.1572.pth", map_location=torch.device('cpu')))
model = model.eval()


champion_rationales = []

# Ensure we have a list of entries
path = "./data/champion_rationales.jsonl"
champion_rationales = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        champion_rationales.append(json.loads(line))    

def get_champion_rationale(champion : str, data : list[dict]):
    for champion_data in data:
        if champion == champion_data["name"]:
            return champion_data["rationale"]



anchor_rationale : str = get_champion_rationale("Sion", champion_rationales)
positive_rationale : str = get_champion_rationale("Ornn", champion_rationales)
negative_rationale : str = get_champion_rationale("Lux", champion_rationales)


anchor_embd : torch.Tensor = model(anchor_rationale)
positive_embd = model(positive_rationale)
negative_embd = model(negative_rationale)

print(anchor_rationale)
print(positive_rationale)
print(negative_rationale)

sim_anchor_pos = cosine_similarity(anchor_embd.detach().cpu().numpy(), positive_embd.detach().cpu().numpy())
sim_anchor_neg = cosine_similarity(anchor_embd.detach().cpu().numpy(), negative_embd.detach().cpu().numpy())

print("Similarity betwen a and p :", sim_anchor_pos)
print("Similarity betwen a and n :", sim_anchor_neg)


    # ensure 1D numpy vector
    # emb = emb.numpy().reshape(-1)

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
import torch
from tqdm import tqdm
from services.huggingface.contrastive.model_loader import PredictionHead
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset

dataset = load_dataset("avinot/Champion-Similarity", split="validation")

model : PredictionHead = PredictionHead(input_dim=2048, output_dim=2048)
model.load_state_dict(torch.load("output/model_epoch_1_step_5001.pth", map_location=torch.device('cpu')))

anchor : str = "Karthus"
positive : str = "Swain"
negative : str = "Lux"

anchor_embedding : torch.Tensor = model(anchor).detach().cpu().to(torch.float32)
positive_embedding : torch.Tensor = model(positive).detach().cpu().to(torch.float32)
negative_embedding : torch.Tensor = model(negative).detach().cpu().to(torch.float32)

sim1 = cosine_similarity(anchor_embedding, positive_embedding)
sim2 = cosine_similarity(anchor_embedding, negative_embedding)
print(f"Similarity between {anchor} and {positive}: {sim1[0][0]}")
print(f"Similarity between {anchor} and {negative}: {sim2[0][0]}")

# good_predictions : int = 0
# bad_predicitons : int = 0
# total_predictions : int = 0

# for example in tqdm(dataset):
#     e_anchor = model(example["anchor"]).detach().cpu().to(torch.float32)
#     e_positive = model(example["positive"]).detach().cpu().to(torch.float32)
#     e_negative = model(example["negative"]).detach().cpu().to(torch.float32)
#     sim1 = cosine_similarity(e_anchor, e_positive)[0][0]
#     sim2 = cosine_similarity(e_anchor, e_negative)[0][0]
    
#     if sim1 > sim2:
#         good_predictions += 1
#     elif sim1 < sim2:
#         bad_predicitons += 1
#     total_predictions += 1
    
# print(f"Good predictions: {good_predictions} ({good_predictions / total_predictions * 100:.2f}%)")
# print(f"Bad predictions: {bad_predicitons} ({bad_predicitons / total_predictions * 100:.2f}%)")
# print(f"Total predictions: {total_predictions}")

# print(anchor_embedding.numpy().tolist())
# print(positive_embedding)
# print(negative_embedding)



import torch
from services.huggingface.contrastive.model_loader import PredictionHead
from sklearn.metrics.pairwise import cosine_similarity

model : PredictionHead = PredictionHead(input_dim=2048, output_dim=2048)

anchor : str = "Zeri"
positive : str = "Miss Fortune"
negative : str = "Sejuani"

anchor_embedding : torch.Tensor = model(anchor)
positive_embedding : torch.Tensor = model(positive)
negative_embedding : torch.Tensor = model(negative)

anchor_embedding = anchor_embedding.detach().cpu().to(torch.float32)
positive_embedding = positive_embedding.detach().cpu().to(torch.float32)
negative_embedding = negative_embedding.detach().cpu().to(torch.float32)

print(anchor_embedding.numpy().tolist())
# print(positive_embedding)
# print(negative_embedding)

loss_fn = torch.nn.TripletMarginLoss(margin=0.5, p=2)
cost = loss_fn(anchor_embedding, positive_embedding, negative_embedding)
print(f"Triplet loss: {cost.item()}")

sim1 = cosine_similarity(anchor_embedding, positive_embedding)
sim2 = cosine_similarity(anchor_embedding, negative_embedding)
print(f"Similarity between anchor and positive: {sim1[0][0]}")
print(f"Similarity between anchor and negative: {sim2[0][0]}")


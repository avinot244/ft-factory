import json
import torch
from tqdm import tqdm
from services.huggingface.contrastive.model_loader import PredictionHead

weights_path = "./output/champion_embedding_v6/best_epoch8_step7271_0.1956.pth"
model = PredictionHead()
model.load_state_dict(torch.load(weights_path))
model.eval()

def get_champion_rationale(champion_name: str, rationale_data : list[dict]) -> str:
    for item in rationale_data:
        if item["name"] == champion_name:
            return item["rationale"]
    return ""

with open("./data/champion_mapping.json", "r") as f:
    champion_mapping = json.load(f)
    champion_list : list[str] = [c["name"] for c in champion_mapping]
    
with open("./data/champion_rationales.jsonl", "r") as f:
    rationale_data : list[dict] = []
    for line in f.readlines():
        rationale_data.append(json.loads(line))

champion_embeddings = {}

with open("./data/champion_embeddings.jsonl", "w") as f:

    for champion_name in tqdm(champion_list[:]):
        rationale = get_champion_rationale(champion_name, rationale_data)
        embedding = model(rationale)
        
        champion_embeddings[champion_name] = embedding.tolist()  # Convert tensor to list for JSON serialization
        
    json.dump(champion_embeddings, f, indent=4)
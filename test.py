from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from utils.token_manager import get_hf_token
from services.huggingface.contrastive.model_loader import PredictionHead

model = PredictionHead(input_dim=2048, output_dim=2048)
# total_params = 2048*2048 + 2048 = 4196352


# Generating embedding for a champion
anchor : str = "Thresh"
positive : str = "Blitzcrank"
negative : str = "Talon"


me_anchor = model(anchor)
me_positive = model(positive)
me_negative = model(negative)

# Computing the loss
loss_fn = torch.nn.TripletMarginLoss(margin=0.1, p=2)
d_ap = torch.dist(me_anchor, me_positive, p=2)
d_an = torch.dist(me_anchor, me_negative, p=2)
print(f"Anchor-Positive distance: {d_ap.item():.4f}")
print(f"Anchor-Negative distance: {d_an.item():.4f}")
print(f"Margin: 0.1, Loss: {max(d_ap - d_an + 0.1, torch.tensor(0.0))}")
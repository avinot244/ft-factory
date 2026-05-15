from sentence_transformers import SentenceTransformer
import torch
from services.huggingface.contrastive.model_loader import PredictionHead
import torch.nn.functional as F

weights_path = "./output/champion_embedding_v6/best_epoch8_step7271_0.1956.pth"
model = PredictionHead()
model.load_state_dict(torch.load(weights_path))
model.eval()

rationale_caitlyn = (
    "The core loop is patient spacing at maximum auto-attack range, "
    "weaving Headshots into ability casts whenever the passive is ready. "
    "Caitlyn demands continuous micro-decisions about trap placement and "
    "positioning — one step too close and the safety advantage disappears. "
    "Players skilled at Kog'Maw or Jinx will recognise the same discipline "
    "of staying untouched to output sustained damage, though Caitlyn's "
    "failure mode is more specific: committing the net or trap before "
    "confirming the enemy has no escape tool, which wastes the only "
    "displacement in the kit."
)

rationale_jinx = (
    "The repeating loop is rocket poke at extended range followed by "
    "minigun cleanup after a takedown triggers the passive reset. "
    "Each fight demands one critical decision — when to switch weapons — "
    "and continuous attack-move micro between those pivots. "
    "Twitch and Kog'Maw share the same stationary hyperscaler discipline, "
    "while Katarina players will recognise the reset-chase pattern. "
    "Jinx's specific failure is switching to minigun before the fight is "
    "won, sacrificing the range that prevents the fed threat from closing."
)

rationale_lux = (
    "The execution pattern is zone establishment via Light Binding placement, "
    "waiting for the enemy to step into the snare, then confirming with "
    "Lucent Singularity and full burst before the root expires. "
    "Lux makes one decision per ability rotation — the rest is scripted "
    "muscle memory once the bind lands. "
    "Veigar and Annie share this zone-placement-then-burst-confirm rhythm "
    "most closely, while the failure mode is throwing the bind predictively "
    "rather than reactively, giving the target reaction time to sidestep "
    "the entire sequence."
)

rationales = {
    "Caitlyn_v6": rationale_caitlyn,
    "Jinx_v6":    rationale_jinx,
    "Lux_v6":     rationale_lux,
}

model = SentenceTransformer("intfloat/e5-base-v2")

embs = model.encode([
    rationale_caitlyn,
    rationale_jinx,
    rationale_lux,
], convert_to_tensor=True, normalize_embeddings=True)

pairs = [("Caitlyn", "Jinx", 0, 1), ("Caitlyn", "Lux", 0, 2), ("Jinx", "Lux", 1, 2)]
for n1, n2, i, j in pairs:
    sim = F.cosine_similarity(embs[i].unsqueeze(0), embs[j].unsqueeze(0)).item()
    print(f"Similarity {n1} / {n2}: {sim:.4f}")
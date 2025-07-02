import torch
import torch.nn as nn
import torch.nn.functional as F

class NCELoss(nn.Module):
    def __init__(self, temperature=0.02):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_emb, pos_emb, neg_embs):
        """
        anchor_emb: [B, D]        - anchor/query vectors
        pos_emb:    [B, D]        - positive vectors
        neg_embs:   [B, D] or [B, K, D] - one or many negatives per query
        """
        B, D = anchor_emb.shape

        # Normalize anchor and positive embeddings
        q = F.normalize(anchor_emb, dim=1)   # [B, D]
        p = F.normalize(pos_emb, dim=1)      # [B, D]

        # Positive similarities: [B]
        sim_pos = torch.exp((q * p).sum(dim=1) / self.temperature)

        # Handle negative embeddings
        if neg_embs.dim() == 2:
            # Single negative per anchor: [B, D]
            n = F.normalize(neg_embs, dim=1)  # [B, D]
            sim_neg = torch.exp((q * n).sum(dim=1) / self.temperature)  # [B]
            denom = sim_pos + sim_neg  # [B]

        elif neg_embs.dim() == 3:
            # Multiple negatives: [B, K, D]
            n = F.normalize(neg_embs, dim=2)  # [B, K, D]
            sim_neg = torch.exp(torch.einsum("bd,bkd->bk", q, n) / self.temperature)  # [B, K]
            denom = sim_pos + sim_neg.sum(dim=1)  # [B]

        else:
            raise ValueError(f"Expected neg_embs to be 2D or 3D, but got shape {neg_embs.shape}")

        # Final loss: -log(sim_pos / denom)
        loss = -torch.log(sim_pos / denom)

        return loss.mean()

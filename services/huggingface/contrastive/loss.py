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

class SIGReg(nn.Module):
    def __init__(self, n_projections: int = 32, lam: float = 0.05):
        super().__init__()
        self.n_projections = n_projections
        self.lam           = lam

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: (B, D) — raw anchor embeddings before normalisation.
        Enforces directional isotropy via characteristic function matching
        on random 1D projections of the L2-normalised embeddings.
        """
        B, D = embeddings.shape
        if B < 4:
            return embeddings.new_tensor(0.0)

        # Normalise onto unit hypersphere — enforces isotropy of direction,
        # not magnitude. This decouples SIGReg from the InfoNCE scale signal.
        emb = F.normalize(embeddings, dim=-1)           # (B, D)

        # Random unit directions sampled fresh each forward pass
        directions = F.normalize(
            torch.randn(
                D,
                self.n_projections,
                device=embeddings.device,
                dtype=emb.dtype,
            ),
            dim=0
        )                                               # (D, M)

        proj = emb @ directions                         # (B, M)

        # Characteristic function matching against N(0,1) target.
        # For each t, E[cos(t*X)] should equal exp(-t²/2) for X ~ N(0,1).
        # We match over a grid of t values for robustness.
        loss = emb.new_tensor(0.0)
        for t in [0.5, 1.0, 1.5, 2.0]:
            ecf_real = torch.cos(t * proj).mean(dim=0)  # (M,)
            target   = torch.exp(emb.new_tensor(-t ** 2 / 2.0))
            loss = loss + (ecf_real - target).pow(2).mean()

        return self.lam * loss
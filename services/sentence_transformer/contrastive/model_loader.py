"""
model_loader.py
---------------
Trainable contrastive bottleneck head over BAAI/bge-m3 backbone embeddings.

Two operating modes, selected at construction time
---------------------------------------------------

CACHED mode  (backbone_name=None, default)
    The SentenceTransformer backbone is NOT loaded.  forward() and
    project_negatives() receive pre-computed float tensors from the
    DataLoader.  Only the tiny MLP head runs — VRAM stays under 512 MB
    even at bs=128.

    model = load_model()                   # or backbone_name=None explicitly
    anc_out = model(anc_emb)               # anc_emb: (B, 1024) Tensor
    neg_out = model.project_negatives(neg_emb, neg_mask)  # (B, K, 1024)

ONLINE mode  (backbone_name="BAAI/bge-m3")
    The SentenceTransformer is loaded and frozen.  forward() and
    encode_negatives() accept raw strings.  The backbone runs on every
    training step — much slower and more VRAM-intensive.

    model = load_model(backbone_name="BAAI/bge-m3")
    anc_out = model(["Zed: ...", "Katarina: ..."])        # list[str]
    neg_out, mask = model.encode_negatives([["Thresh: ...", ""], ...])

Architecture (both modes)
--------------------------

    input  (B, 1024) tensor  [cached]
    OR
    list[str] → SentenceTransformer → (B, 1024)  [online]
          │
          ▼
    _BottleneckMLP   (trainable)
          │  Linear(1024→hidden_dim) + LayerNorm + GELU + Dropout
          │  Linear(hidden_dim→bottleneck_dim) + LayerNorm + GELU + Dropout
          │  Linear(bottleneck_dim→output_dim)
          ▼
    L2-normalised embedding  (B, output_dim)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_DIM   = 1024   # bge-m3 dense output dimension
_PAD_SENTINEL  = ""     # must match ContrastiveCollator._PAD  (online mode)


# ---------------------------------------------------------------------------
# Bottleneck MLP
# ---------------------------------------------------------------------------

class _BottleneckMLP(nn.Module):
    """
    Three-layer MLP:  input_dim → hidden_dim → bottleneck_dim → output_dim

    Each Linear except the last is followed by LayerNorm + GELU + Dropout.
    The caller applies L2 normalisation to the final output.
    """

    def __init__(
        self,
        input_dim:      int,
        hidden_dim:     int,
        bottleneck_dim: int,
        output_dim:     int,
        dropout:        float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim,      hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim,     bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# PredictionHead
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """
    Contrastive projection head that works in CACHED or ONLINE mode.

    Parameters
    ----------
    output_dim     : final embedding dimensionality (L2-normalised)
    hidden_dim     : first MLP projection size
    bottleneck_dim : compressed bottleneck size
    dropout        : dropout rate inside the MLP
    backbone_name  : if None → CACHED mode (tensor input, no ST loaded).
                     if str  → ONLINE mode (string input, ST backbone frozen).
    device         : torch device string (used only in ONLINE mode for the ST)
    """

    def __init__(
        self,
        output_dim:     int            = 256,
        hidden_dim:     int            = 512,
        bottleneck_dim: int            = 256,
        dropout:        float          = 0.1,
        backbone_name:  Optional[str]  = None,
        device:         str            = "cpu",
    ) -> None:
        super().__init__()

        self.backbone_name = backbone_name

        # -- Optional backbone (ONLINE mode only) ---------------------------
        self.backbone = None
        if backbone_name is not None:
            from sentence_transformers import SentenceTransformer
            self.backbone = SentenceTransformer(backbone_name, device=device)
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[PredictionHead] ONLINE mode — backbone '{backbone_name}' loaded and frozen.")
        else:
            print("[PredictionHead] CACHED mode — no backbone loaded; expects pre-computed tensors.")

        # -- Trainable head (both modes) ------------------------------------
        self.head = _BottleneckMLP(
            input_dim      = BACKBONE_DIM,
            hidden_dim     = hidden_dim,
            bottleneck_dim = bottleneck_dim,
            output_dim     = output_dim,
            dropout        = dropout,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def cached(self) -> bool:
        """True when operating in CACHED (tensor-input) mode."""
        return self.backbone is None

    def _project(self, backbone_emb: torch.Tensor) -> torch.Tensor:
        """Run the MLP head + L2 normalisation on a (N, 1024) tensor."""
        return F.normalize(self.head(backbone_emb), dim=1)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """ONLINE mode only: encode a flat list of strings via the ST backbone."""
        assert self.backbone is not None, \
            "encode_texts called in CACHED mode — pass a backbone_name to enable ONLINE mode."
        emb = self.backbone.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return emb.to(next(self.head.parameters()).device)

    # ------------------------------------------------------------------
    # Public forward  (CACHED mode)
    # ------------------------------------------------------------------

    def forward(self, backbone_emb: torch.Tensor) -> torch.Tensor:
        """
        CACHED mode: project a batch of pre-computed backbone embeddings.

        Parameters
        ----------
        backbone_emb : FloatTensor  (B, 1024)

        Returns
        -------
        FloatTensor  (B, output_dim)  — L2-normalised
        """
        return self._project(backbone_emb)

    def project_negatives(
        self,
        neg_emb:  torch.Tensor,      # (B, K, 1024)
        neg_mask: torch.BoolTensor,  # (B, K)
    ) -> torch.Tensor:
        """
        CACHED mode: project a padded block of negative backbone embeddings.

        Only valid (non-padding) positions are passed through the MLP;
        padding positions remain zero vectors in the output.

        Returns
        -------
        FloatTensor  (B, K, output_dim)
        """
        B, K, _ = neg_emb.shape
        output_dim = self.head.net[-1].out_features
        out = torch.zeros(B, K, output_dim, device=neg_emb.device)
        if neg_mask.any():
            out[neg_mask] = self._project(neg_emb[neg_mask])   # (M, output_dim)
        return out

    # ------------------------------------------------------------------
    # Public forward  (ONLINE mode)
    # ------------------------------------------------------------------

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        ONLINE mode: encode a batch of strings end-to-end.

        Parameters
        ----------
        texts : list[str]  length B

        Returns
        -------
        FloatTensor  (B, output_dim)  — L2-normalised
        """
        return self._project(self._encode_texts(texts))

    def encode_negatives(
        self,
        negative_texts: list[list[str]],
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        ONLINE mode: encode a padded batch of negative string lists.

        Parameters
        ----------
        negative_texts : (B, K_max)  — padding positions contain ""

        Returns
        -------
        neg_emb  : FloatTensor   (B, K_max, output_dim)
        neg_mask : BoolTensor    (B, K_max)  True = real negative
        """
        B     = len(negative_texts)
        K_max = max(len(row) for row in negative_texts)
        device = next(self.head.parameters()).device

        mask = torch.tensor(
            [[t != _PAD_SENTINEL for t in row] for row in negative_texts],
            dtype=torch.bool, device=device,
        )

        flat_texts   = [t for row in negative_texts for t in row if t != _PAD_SENTINEL]
        flat_indices = [(b, k) for b, row in enumerate(negative_texts)
                        for k, t in enumerate(row) if t != _PAD_SENTINEL]

        output_dim = self.head.net[-1].out_features
        neg_emb    = torch.zeros(B, K_max, output_dim, device=device)

        if flat_texts:
            flat_out = self.encode(flat_texts)              # (M, output_dim)
            for i, (b, k) in enumerate(flat_indices):
                neg_emb[b, k] = flat_out[i]

        return neg_emb, mask


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(
    output_dim:      int           = 256,
    hidden_dim:      int           = 512,
    bottleneck_dim:  int           = 256,
    dropout:         float         = 0.1,
    backbone_name:   Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device:          str           = "cpu",
) -> PredictionHead:
    """
    Instantiate (and optionally restore) a PredictionHead.

    Parameters
    ----------
    backbone_name
        None  → CACHED mode (default, recommended for training).
        "BAAI/bge-m3" or any ST model → ONLINE mode.
    checkpoint_path
        If given, restore head weights from this .pth state-dict file.
    device
        Torch device string.

    Returns
    -------
    PredictionHead on `device` in eval mode.
    """
    model = PredictionHead(
        output_dim     = output_dim,
        hidden_dim     = hidden_dim,
        bottleneck_dim = bottleneck_dim,
        dropout        = dropout,
        backbone_name  = backbone_name,
        device         = device,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        print(f"[model_loader] Restored checkpoint: {checkpoint_path}")

    return model.to(device).eval()
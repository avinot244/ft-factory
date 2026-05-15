"""
model_loader.py
---------------
Loads BAAI/bge-m3 as a frozen sentence-transformer backbone and attaches a
trainable contrastive bottleneck head on top.

Architecture
------------

    text(s)
      │
      ▼
  BAAI/bge-m3  (SentenceTransformer — backbone, frozen by default)
      │  CLS-pooled dense vector  (1024-d)
      ▼
  PredictionHead  (trainable MLP bottleneck)
      │  projection  →  bottleneck  →  projection
      ▼
  embedding  (output_dim-d, L2-normalised)

The backbone is loaded via the `sentence-transformers` library so pooling
and normalisation layers that are part of bge-m3's standard pipeline are
respected.  Only the head is updated during contrastive training.

Usage
-----
    from model_loader import PredictionHead, load_model

    model = load_model(output_dim=256, freeze_backbone=True)
    model = model.to(device)

    # Single forward pass (returns L2-normalised embeddings)
    emb = model(["Zed: ...", "Katarina: ..."])   # (2, 256)

    # Negative forward pass (handles padding mask)
    neg_emb, neg_mask = model.encode_negatives([
        ["Thresh: ...", "Braum: ...", ""],
        ["Anivia: ...", "", ""],
    ])  # neg_emb: (2, 3, 256),  neg_mask: (2, 3) bool
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BACKBONE_NAME  = "BAAI/bge-m3"
BACKBONE_DIM   = 1024   # bge-m3 dense output dimension
_PAD_SENTINEL  = ""     # must match ContrastiveCollator._PAD


# ---------------------------------------------------------------------------
# Contrastive bottleneck head
# ---------------------------------------------------------------------------

class _BottleneckMLP(nn.Module):
    """
    Three-layer MLP:  input_dim → hidden_dim → bottleneck_dim → output_dim

    Each Linear is followed by LayerNorm + GELU, except the last projection
    which has no activation (the caller applies L2 normalisation instead).
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
# Full model
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """
    Frozen BAAI/bge-m3 backbone  +  trainable contrastive bottleneck head.

    Parameters
    ----------
    output_dim      : dimensionality of the final L2-normalised embedding
    hidden_dim      : first projection in the bottleneck MLP
    bottleneck_dim  : compressed representation before the final projection
    dropout         : dropout applied inside the MLP
    freeze_backbone : if True (default) backbone weights are frozen
    device          : torch device string, e.g. "cuda" or "cpu"
    """

    def __init__(
        self,
        output_dim:      int   = 256,
        hidden_dim:      int   = 512,
        bottleneck_dim:  int   = 256,
        dropout:         float = 0.1,
        freeze_backbone: bool  = True,
        device:          str   = torch.cuda.is_available() and "cuda" or "cpu",
    ) -> None:
        super().__init__()

        self.device_str = device

        # -- Backbone -------------------------------------------------------
        self.backbone = SentenceTransformer(BACKBONE_NAME, device=device)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # -- Contrastive head -----------------------------------------------
        self.head = _BottleneckMLP(
            input_dim      = BACKBONE_DIM,
            hidden_dim     = hidden_dim,
            bottleneck_dim = bottleneck_dim,
            output_dim     = output_dim,
            dropout        = dropout,
        )

    # ------------------------------------------------------------------
    # Internal encoding
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> torch.Tensor:
        """
        Run bge-m3 on a flat list of strings, return CLS-pooled vectors
        as a torch.Tensor on the model's device.

        Shape: (N, BACKBONE_DIM)
        """
        # sentence-transformers returns numpy arrays by default; ask for tensors
        emb = self.backbone.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=False,   # we normalise after the head
            show_progress_bar=False,
        )
        return emb.to(next(self.head.parameters()).device)

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a batch of strings and project through the contrastive head.

        Parameters
        ----------
        texts : list[str]  length B

        Returns
        -------
        torch.Tensor  (B, output_dim)  — L2-normalised
        """
        backbone_emb = self._encode(texts)          # (B, BACKBONE_DIM)
        projected    = self.head(backbone_emb)       # (B, output_dim)
        return F.normalize(projected, dim=1)         # (B, output_dim)

    def encode_negatives(
        self,
        negative_texts: list[list[str]],
    ) -> tuple[torch.Tensor, torch.BoolTensor]:
        """
        Encode a padded batch of negative lists.

        Parameters
        ----------
        negative_texts : (B, K_max)  — padding positions contain _PAD_SENTINEL

        Returns
        -------
        neg_emb  : FloatTensor  (B, K_max, output_dim)  — L2-normalised
                   padding positions are filled with zeros
        neg_mask : BoolTensor   (B, K_max)  True = real negative
        """
        B      = len(negative_texts)
        K_max  = max(len(row) for row in negative_texts)
        device = next(self.head.parameters()).device

        # Build mask
        mask = torch.tensor(
            [[text != _PAD_SENTINEL for text in row] for row in negative_texts],
            dtype=torch.bool,
            device=device,
        )                                            # (B, K_max)

        # Flatten valid texts only
        flat_texts  = [
            text
            for row in negative_texts
            for text in row
            if text != _PAD_SENTINEL
        ]
        flat_indices = [
            (b, k)
            for b, row in enumerate(negative_texts)
            for k, text in enumerate(row)
            if text != _PAD_SENTINEL
        ]

        # Allocate output — zeros for padding positions
        output_dim = self.head.net[-1].out_features
        neg_emb    = torch.zeros(B, K_max, output_dim, device=device)

        if flat_texts:
            flat_emb = self.forward(flat_texts)     # (M, output_dim)
            for emb_i, (b, k) in enumerate(flat_indices):
                neg_emb[b, k] = flat_emb[emb_i]

        return neg_emb, mask


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def load_model(
    output_dim:      int   = 256,
    hidden_dim:      int   = 512,
    bottleneck_dim:  int   = 256,
    dropout:         float = 0.1,
    freeze_backbone: bool  = True,
    checkpoint_path: Optional[str] = None,
    device:          str   = "cpu",
) -> PredictionHead:
    """
    Instantiate (and optionally restore) a PredictionHead.

    Parameters
    ----------
    output_dim, hidden_dim, bottleneck_dim, dropout, freeze_backbone
        Architecture hyper-parameters (see PredictionHead).
    checkpoint_path
        If given, load head weights from this .pth file (state_dict only).
    device
        Torch device string.

    Returns
    -------
    PredictionHead (on `device`, in eval mode)
    """
    model = PredictionHead(
        output_dim      = output_dim,
        hidden_dim      = hidden_dim,
        bottleneck_dim  = bottleneck_dim,
        dropout         = dropout,
        freeze_backbone = freeze_backbone,
        device          = device,
    )

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        print(f"[model_loader] Loaded checkpoint: {checkpoint_path}")

    return model.to(device).eval()
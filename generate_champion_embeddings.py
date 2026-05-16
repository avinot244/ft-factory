"""
generate_embeddings.py
----------------------
Generate final champion embeddings from a JSONL file of champion rationales,
using a trained PredictionHead and the pre-built BAAI/bge-m3 backbone cache.

Input JSONL format (one JSON object per line)
---------------------------------------------
    {"name": "Zed", "rationale": "An energy-based ..."}
    {"name": "Katarina", "rationale": "A high-mobility ..."}

Output
------
A single .json file mapping champion names to their L2-normalised
    embedding vectors:
    {
        "Zed":      [0.12, -0.03, ...],
        "Katarina": [...],
        ...
    }

Two modes, automatically selected
-----------------------------------
CACHED mode  (cache_path is provided and the file exists)
    Backbone embeddings are looked up from the pre-built cache dict.
    The SentenceTransformer is never loaded.  Fast, CPU-friendly.

ONLINE mode  (cache_path=None or file missing)
    Rationales are encoded on-the-fly by BAAI/bge-m3.
    Requires sentence-transformers and more VRAM.

Usage
-----
    # CACHED (recommended)
    python generate_embeddings.py \\
        --rationales  data/champion_rationales.jsonl \\
        --checkpoint  output/c2cp_bge/best.pth \\
        --cache_path  cache/backbone_cache.pt \\
        --output      embeddings/champion_embeddings.pt

    # ONLINE (no cache)
    python generate_embeddings.py \\
        --rationales  data/champion_rationales.jsonl \\
        --checkpoint  output/c2cp_bge/best.pth \\
        --output      embeddings/champion_embeddings.pt

    # From Python
    from generate_embeddings import generate_embeddings
    result = generate_embeddings(
        rationales_path = "data/champion_rationales.jsonl",
        checkpoint_path = "output/c2cp_bge/best.pth",
        cache_path      = "cache/backbone_cache.pt",
        output_path     = "embeddings/champion_embeddings.pt",
    )
    # result["embeddings"] : (N, output_dim) FloatTensor
    # result["names"]      : list[str]
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import torch
import torch.nn.functional as F

from services.sentence_transformer.contrastive.model_loader import PredictionHead, load_model


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def _load_rationales(path: str) -> tuple[list[str], list[str]]:
    """
    Parse the input JSONL file.

    Returns
    -------
    names      : list[str]  champion names
    rationales : list[str]  corresponding rationale strings
    """
    names, rationales = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i + 1} of '{path}': {e}")
            if "name" not in obj or "rationale" not in obj:
                raise ValueError(
                    f"Line {i + 1} is missing 'name' or 'rationale' key: {obj}"
                )
            names.append(obj["name"])
            rationales.append(obj["rationale"])

    if not names:
        raise ValueError(f"No entries found in '{path}'.")

    print(f"[generate] Loaded {len(names)} champions from '{path}'.")
    return names, rationales


# ---------------------------------------------------------------------------
# Backbone lookup (CACHED mode)
# ---------------------------------------------------------------------------

def _lookup_from_cache(
    rationales: list[str],
    cache_path: str,
) -> torch.Tensor:
    """
    Look up pre-computed backbone embeddings for each rationale.

    Parameters
    ----------
    rationales : list[str]  length N
    cache_path : path to the backbone_cache.pt file

    Returns
    -------
    FloatTensor  (N, BACKBONE_DIM)  on CPU
    """
    print(f"[generate] Loading backbone cache from '{cache_path}'...")
    cache: dict[str, torch.Tensor] = torch.load(
        cache_path, map_location="cpu", weights_only=True
    )
    print(f"[generate] Cache contains {len(cache)} entries.")

    missing = [r for r in rationales if r not in cache]
    if missing:
        raise KeyError(
            f"{len(missing)} rationale(s) not found in cache.\n"
            f"First missing: {missing[0][:120]!r}\n"
            f"Re-run build_cache() to include these champions."
        )

    backbone_embs = torch.stack([cache[r] for r in rationales])  # (N, D)
    print(f"[generate] Cache lookup complete — shape {tuple(backbone_embs.shape)}.")
    return backbone_embs


# ---------------------------------------------------------------------------
# Backbone encoding (ONLINE mode)
# ---------------------------------------------------------------------------

def _encode_online(
    rationales: list[str],
    backbone_name: str,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """
    Encode rationales with the ST backbone (no cache).

    Returns
    -------
    FloatTensor  (N, BACKBONE_DIM)  on CPU
    """
    from sentence_transformers import SentenceTransformer

    print(f"[generate] ONLINE mode — loading '{backbone_name}'...")
    backbone = SentenceTransformer(backbone_name, device=device)
    backbone.eval()

    with torch.no_grad():
        embs = backbone.encode(
            rationales,
            batch_size=batch_size,
            convert_to_tensor=True,
            normalize_embeddings=False,
            show_progress_bar=True,
        )
    return embs.cpu().float()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_embeddings(
    rationales_path: str,
    checkpoint_path: str,
    output_path:     str,
    cache_path:      Optional[str] = None,
    backbone_name:   str           = "BAAI/bge-m3",
    output_dim:      int           = 256,
    hidden_dim:      int           = 512,
    bottleneck_dim:  int           = 256,
    dropout:         float         = 0.0,   # eval — no dropout
    device:          str           = "cpu",
    batch_size:      int           = 64,    # used in ONLINE mode only
) -> dict[str, list[float]]:
    """
    Generate L2-normalised champion embeddings from a JSONL rationale file.

    Parameters
    ----------
    rationales_path : path to the input .jsonl file
    checkpoint_path : path to the trained PredictionHead .pth checkpoint
    output_path     : where to save the output .pt file
    cache_path      : path to backbone_cache.pt.
                      If provided and the file exists → CACHED mode.
                      If None or file missing → ONLINE mode.
    backbone_name   : ST model name used in ONLINE mode (ignored in CACHED mode)
    output_dim, hidden_dim, bottleneck_dim, dropout
                    : must match the architecture used during training
    device          : torch device for the MLP head ("cpu" is fine for inference)
    batch_size      : mini-batch size for ONLINE backbone encoding

    Returns
    -------
    dict mapping champion name → list[float] (the L2-normalised embedding)
    """
    # -- 1. Load rationales -------------------------------------------------
    names, rationales = _load_rationales(rationales_path)
    N = len(names)

    # -- 2. Get backbone embeddings (cached or online) ----------------------
    use_cache = cache_path is not None and os.path.exists(cache_path)

    if use_cache:
        print("[generate] Mode: CACHED")
        backbone_embs = _lookup_from_cache(rationales, cache_path)
    else:
        if cache_path is not None:
            print(f"[generate] Cache file not found at '{cache_path}' — falling back to ONLINE mode.")
        else:
            print("[generate] Mode: ONLINE")
        backbone_embs = _encode_online(rationales, backbone_name, batch_size, device)

    # backbone_embs: (N, BACKBONE_DIM) on CPU

    # -- 3. Load the trained MLP head (no backbone) -------------------------
    print(f"[generate] Loading PredictionHead from '{checkpoint_path}'...")
    model = load_model(
        output_dim     = output_dim,
        hidden_dim     = hidden_dim,
        bottleneck_dim = bottleneck_dim,
        dropout        = dropout,
        backbone_name  = None,             # always tensor-mode for inference
        checkpoint_path = checkpoint_path,
        device         = device,
    )
    model.eval()

    # -- 4. Project through the head in batches -----------------------------
    print(f"[generate] Projecting {N} embeddings through the head (batch_size={batch_size})...")
    all_projected: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end   = min(start + batch_size, N)
            batch = backbone_embs[start:end].to(device)  # (B, BACKBONE_DIM)
            proj  = model(batch)                          # (B, output_dim)
            all_projected.append(proj.cpu())

    embeddings = torch.cat(all_projected, dim=0)          # (N, output_dim)
    assert embeddings.shape == (N, output_dim), \
        f"Shape mismatch: expected ({N}, {output_dim}), got {tuple(embeddings.shape)}"

    # -- 5. Save output -----------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    result = {
        name: embeddings[i].tolist()
        for i, name in enumerate(names)
    }
    with open(output_path, "w") as f:
        json.dump(result, f)
    print(f"[generate] Saved {N} embeddings (dim={output_dim}) → '{output_path}'")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate champion embeddings from a trained PredictionHead."
    )
    p.add_argument("--rationales",    required=True,
                   help="Path to input .jsonl file (name + rationale per line).")
    p.add_argument("--checkpoint",    required=True,
                   help="Path to trained PredictionHead .pth checkpoint.")
    p.add_argument("--output",        required=True,
                   help="Destination .pt file for the output embeddings.")
    p.add_argument("--cache_path",    default=None,
                   help="Path to backbone_cache.pt (CACHED mode). "
                        "Omit to encode rationales online with bge-m3.")
    p.add_argument("--backbone_name", default="BAAI/bge-m3",
                   help="ST backbone name for ONLINE mode (default: BAAI/bge-m3).")
    p.add_argument("--output_dim",    type=int,   default=256)
    p.add_argument("--hidden_dim",    type=int,   default=512)
    p.add_argument("--bottleneck_dim",type=int,   default=256)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--batch_size",    type=int,   default=64,
                   help="Batch size for head projection (and backbone encoding in ONLINE mode).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_embeddings(
        rationales_path = args.rationales,
        checkpoint_path = args.checkpoint,
        output_path     = args.output,
        cache_path      = args.cache_path,
        backbone_name   = args.backbone_name,
        output_dim      = args.output_dim,
        hidden_dim      = args.hidden_dim,
        bottleneck_dim  = args.bottleneck_dim,
        device          = args.device,
        batch_size      = args.batch_size,
    )
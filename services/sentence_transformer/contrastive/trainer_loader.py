"""
trainer_loader.py
-----------------
Contrastive trainer for PredictionHead, supporting both operating modes:

CACHED mode  (args.cache_path is not None)
    DataLoader delivers pre-computed (B, 1024) backbone tensors.
    Only the MLP head runs per step — VRAM < 512 MB at bs=128.
    Recommended for all training runs.

ONLINE mode  (args.cache_path is None, args.backbone_name is not None)
    DataLoader delivers raw strings.  The frozen ST backbone encodes them
    on every step.  Slower and more VRAM-intensive (~4 GB at bs=16).

The mode is selected entirely through ContrastiveTrainingArgs:

    # CACHED (recommended)
    run(ContrastiveTrainingArgs(cache_path="cache/backbone_cache.pt"))

    # ONLINE (no cache)
    run(ContrastiveTrainingArgs(cache_path=None, backbone_name="BAAI/bge-m3"))

Tracked metrics (logged at every logging_steps / eval_steps):
  type == "train" | "val":
    global_step, epoch, loss,
    cos_pos, cos_neg_mean, cos_neg_std, gap_mean, gap_std,
    eff_rank, per_dim_std, mpcs
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from services.sentence_transformer.contrastive.data_loader import (
    ChampionSimilarityDataset,
    build_dataloader,
)
from services.sentence_transformer.contrastive.model_loader import PredictionHead, load_model


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

@dataclass
class ContrastiveTrainingArgs:
    # Paths
    output_dir:    str           = "output/c2cp_bge"
    logging_path:  str           = "logs/c2cp_bge/c2cp_bge_metrics.jsonl"
    # Set to a .pt path for CACHED mode; None for ONLINE mode
    cache_path:    Optional[str] = "cache/backbone_cache.pt"
    # Used only in ONLINE mode (cache_path=None)
    backbone_name: Optional[str] = "BAAI/bge-m3"

    # Training hyper-parameters
    epochs:           int   = 10
    train_batch_size: int   = 64
    eval_batch_size:  int   = 128
    learning_rate:    float = 3e-4
    weight_decay:     float = 1e-2

    # Loss hyper-parameters
    temperature:  float = 0.07
    margin:       float = 0.5
    p:            int   = 2
    use_sigreg:   bool  = True

    # Logging / checkpointing
    logging_steps: int = 20
    eval_steps:    int = 100

    # Model architecture
    output_dim:     int   = 256
    hidden_dim:     int   = 512
    bottleneck_dim: int   = 256
    dropout:        float = 0.1

    # HuggingFace dataset cache
    hf_cache_dir: Optional[str] = None

    # Device
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.cache_path is None and self.backbone_name is None:
            raise ValueError(
                "At least one of cache_path or backbone_name must be set.\n"
                "  CACHED mode : set cache_path='path/to/backbone_cache.pt'\n"
                "  ONLINE mode : set backbone_name='BAAI/bge-m3' (and cache_path=None)"
            )

    @property
    def cached(self) -> bool:
        """True when running in CACHED (tensor) mode."""
        return self.cache_path is not None


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class NCELoss(torch.nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor:   torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        neg_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        B       = anchor.size(0)
        pos_sim = (anchor * positive).sum(dim=1, keepdim=True) / self.temperature
        neg_sim = torch.einsum("bd,bkd->bk", anchor, negative) / self.temperature
        neg_sim = neg_sim.masked_fill(~neg_mask, float("-inf"))
        logits  = torch.cat([pos_sim, neg_sim], dim=1)
        labels  = torch.zeros(B, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


class SIGReg(torch.nn.Module):
    def __init__(self, n_projections: int = 32, lam: float = 0.05) -> None:
        super().__init__()
        self.n_projections = n_projections
        self.lam           = lam

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        B, D = embeddings.shape
        W    = F.normalize(torch.randn(D, self.n_projections, device=embeddings.device), dim=0)
        proj = embeddings @ W
        diff = proj.unsqueeze(0) - proj.unsqueeze(1)
        sq   = (diff ** 2).sum(dim=2)
        return self.lam * torch.log(torch.exp(-2.0 * sq).mean())


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _effective_rank(embeddings: torch.Tensor) -> float:
    with torch.no_grad():
        e = embeddings.float()
        _, s, _ = torch.linalg.svd(e, full_matrices=False)
        s = s[s > 0]; p = s / s.sum()
        return float(torch.exp(-(p * torch.log(p)).sum()).item())


def _mean_pairwise_cos(embeddings: torch.Tensor) -> float:
    with torch.no_grad():
        e = F.normalize(embeddings.float(), dim=1)
        sim = e @ e.T; N = sim.size(0)
        if N < 2:
            return float(sim.mean().item())
        idx = torch.triu_indices(N, N, offset=1)
        return float(sim[idx[0], idx[1]].mean().item())


def _compute_metrics(anc, pos, neg, neg_mask) -> dict:
    with torch.no_grad():
        anc_n  = F.normalize(anc.float(), dim=1)
        pos_n  = F.normalize(pos.float(), dim=1)
        cp     = (anc_n * pos_n).sum(dim=1)
        neg_n  = F.normalize(neg.float(), dim=2)
        cn     = torch.einsum("bd,bkd->bk", anc_n, neg_n)
        valid  = cn[neg_mask]
        cp_exp = cp.unsqueeze(1).expand_as(cn)
        gaps   = (cp_exp - cn)[neg_mask]
    return {
        "cos_pos":      float(cp.mean().item()),
        "cos_neg_mean": float(valid.mean().item()),
        "cos_neg_std":  float(valid.std().item()),
        "gap_mean":     float(gaps.mean().item()),
        "gap_std":      float(gaps.std().item()),
        "eff_rank":     _effective_rank(anc),
        "per_dim_std":  float(anc.float().std(dim=0).mean().item()),
        "mpcs":         _mean_pairwise_cos(anc),
    }


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------

class MetricLogger:
    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(self.path, "w") as _:
            pass

    def _write(self, row: dict) -> None:
        with open(self.path, "a") as f:
            json.dump(row, f); f.write("\n")

    def log_train(self, global_step, epoch, loss, anc, pos, neg, neg_mask) -> dict:
        m = _compute_metrics(anc, pos, neg, neg_mask)
        self._write({"type": "train", "global_step": global_step,
                     "epoch": epoch, "loss": loss, **m})
        return m

    def log_val(self, global_step, epoch, loss, anc, pos, neg, neg_mask) -> dict:
        m = _compute_metrics(anc, pos, neg, neg_mask)
        self._write({"type": "val", "global_step": global_step,
                     "epoch": epoch, "loss": loss, **m})
        return m


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> tuple[dict, dict]:
    train: dict[str, list] = {}
    val:   dict[str, list] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            row  = json.loads(line)
            dest = train if row["type"] == "train" else val
            for k, v in row.items():
                if k == "type": continue
                dest.setdefault(k, []).append(v)
    return train, val


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_STYLE = {
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.35, "lines.linewidth": 1.6,
}
_TRAIN = "#0072B2"
_VAL   = "#D55E00"
_NEG   = "#E69F00"


def _new_fig(title):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.set_title(title, pad=8); return fig, ax


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def _band(ax, steps, mean_vals, std_vals, color, alpha=0.16, **kw):
    m, s = np.asarray(mean_vals), np.asarray(std_vals)
    ax.plot(steps, m, color=color, **kw)
    ax.fill_between(steps, m - s, m + s, color=color, alpha=alpha)


def plot_all(train: dict, val: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    has_val = bool(val.get("global_step"))

    with plt.rc_context(_STYLE):
        fig, ax = _new_fig("Training & Validation Loss")
        ax.plot(train["global_step"], train["loss"], color=_TRAIN, label="Train")
        if has_val: ax.plot(val["global_step"], val["loss"], color=_VAL, linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Loss"); ax.legend()
        _save(fig, os.path.join(output_dir, "01_loss.png"))

        fig, ax = _new_fig("Anchor-Positive Cosine Similarity")
        ax.plot(train["global_step"], train["cos_pos"], color=_TRAIN, label="Train")
        if has_val: ax.plot(val["global_step"], val["cos_pos"], color=_VAL, linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Cosine similarity"); ax.legend()
        _save(fig, os.path.join(output_dir, "02_cos_pos.png"))

        fig, ax = _new_fig("Anchor-Negative Cosine Similarity  (mean ± 1 std)")
        _band(ax, train["global_step"], train["cos_neg_mean"], train["cos_neg_std"], _NEG, label="Train mean")
        if has_val: _band(ax, val["global_step"], val["cos_neg_mean"], val["cos_neg_std"], _VAL, linestyle="--", label="Val mean")
        ax.set_xlabel("Global step"); ax.set_ylabel("Cosine similarity"); ax.legend()
        _save(fig, os.path.join(output_dir, "03_cos_neg.png"))

        fig, ax = _new_fig("Similarity Gap  cos(a,p) − cos(a,n)  (mean ± 1 std)")
        _band(ax, train["global_step"], train["gap_mean"], train["gap_std"], _TRAIN, label="Train mean")
        if has_val: _band(ax, val["global_step"], val["gap_mean"], val["gap_std"], _VAL, linestyle="--", label="Val mean")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Global step"); ax.set_ylabel("Similarity gap"); ax.legend()
        _save(fig, os.path.join(output_dir, "04_gap.png"))

        fig, ax = _new_fig("Embedding Effective Rank  (Roy & Vetterli, 2007)")
        ax.plot(train["global_step"], train["eff_rank"], color=_TRAIN, label="Train")
        if has_val: ax.plot(val["global_step"], val["eff_rank"], color=_VAL, linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Effective rank"); ax.legend()
        _save(fig, os.path.join(output_dir, "05_effective_rank.png"))

        fig, ax = _new_fig("Mean Per-Dimension Std of Anchor Embeddings")
        ax.plot(train["global_step"], train["per_dim_std"], color=_TRAIN, label="Train")
        if has_val: ax.plot(val["global_step"], val["per_dim_std"], color=_VAL, linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Mean per-dim std"); ax.legend()
        _save(fig, os.path.join(output_dir, "06_per_dim_std.png"))

        fig, ax = _new_fig("Global Mean Pairwise Cosine Similarity  (batch anchors)")
        ax.plot(train["global_step"], train["mpcs"], color=_TRAIN, label="Train")
        if has_val: ax.plot(val["global_step"], val["mpcs"], color=_VAL, linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Mean pairwise cosine sim"); ax.legend()
        _save(fig, os.path.join(output_dir, "07_mean_pairwise_cos.png"))

        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        for ax, (title, key) in zip(axes, [
            ("Effective Rank", "eff_rank"), ("Per-Dim Std", "per_dim_std"), ("Mean Pairwise Cos", "mpcs"),
        ]):
            ax.set_title(title, fontsize=11)
            ax.plot(train["global_step"], train[key], color=_TRAIN, label="Train")
            if has_val: ax.plot(val["global_step"], val[key], color=_VAL, linestyle="--", label="Val")
            ax.set_xlabel("Step")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.35); ax.legend(fontsize=8)
        fig.suptitle("Representation Collapse Diagnostics", fontsize=12, y=1.01)
        _save(fig, os.path.join(output_dir, "08_collapse_overview.png"))

    print(f"[Plotter] 8 figures saved to {output_dir}")


def plot_from_jsonl(jsonl_path: str, output_dir: str) -> None:
    train, val = load_jsonl(jsonl_path)
    plot_all(train, val, output_dir)


# ---------------------------------------------------------------------------
# Forward helpers  (mode-dispatched, used by both train loop and eval pass)
# ---------------------------------------------------------------------------

def _forward_batch(
    model:    PredictionHead,
    batch:    tuple,
    device:   str,
    cached:   bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
    """
    Dispatch one DataLoader batch through the model for either mode.

    CACHED : batch = (anc_emb, pos_emb, neg_emb, neg_mask)  — all tensors
    ONLINE : batch = (anc_texts, pos_texts, neg_texts, neg_mask)  — strings + mask

    Returns
    -------
    anc_out  : (B, D)
    pos_out  : (B, D)
    neg_out  : (B, K, D)
    neg_mask : (B, K) BoolTensor  on device
    """
    if cached:
        anc_emb, pos_emb, neg_emb, neg_mask = batch
        anc_emb  = anc_emb.to(device)
        pos_emb  = pos_emb.to(device)
        neg_emb  = neg_emb.to(device)
        neg_mask = neg_mask.to(device)
        anc_out  = model(anc_emb)
        pos_out  = model(pos_emb)
        neg_out  = model.project_negatives(neg_emb, neg_mask)
    else:
        anc_texts, pos_texts, neg_texts, neg_mask = batch
        neg_mask = neg_mask.to(device)
        anc_out  = model.encode(anc_texts)
        pos_out  = model.encode(pos_texts)
        neg_out, neg_mask = model.encode_negatives(neg_texts)   # mask rebuilt inside

    return anc_out, pos_out, neg_out, neg_mask


def _compute_loss(
    anc_out:    torch.Tensor,
    pos_out:    torch.Tensor,
    neg_out:    torch.Tensor,
    neg_mask:   torch.BoolTensor,
    nce_loss:   NCELoss,
    sigreg_loss: Optional[SIGReg],
    args:       ContrastiveTrainingArgs,
) -> torch.Tensor:
    K       = neg_out.size(1)
    anc_rep = anc_out.unsqueeze(1).expand(-1, K, -1)
    pos_rep = pos_out.unsqueeze(1).expand(-1, K, -1)

    triplet = F.relu(
        F.pairwise_distance(anc_rep[neg_mask], pos_rep[neg_mask], p=args.p)
        - F.pairwise_distance(anc_rep[neg_mask], neg_out[neg_mask], p=args.p)
        + args.margin
    ).mean()

    cost = nce_loss(anc_out, pos_out, neg_out, neg_mask) + 0.5 * triplet
    if args.use_sigreg and sigreg_loss is not None:
        cost = cost + 0.5 * (sigreg_loss(anc_out) + sigreg_loss(pos_out))
    return cost


# ---------------------------------------------------------------------------
# Eval helper
# ---------------------------------------------------------------------------

def _eval_pass(
    dataloader_val: torch.utils.data.DataLoader,
    model:          PredictionHead,
    nce_loss:       NCELoss,
    args:           ContrastiveTrainingArgs,
    sigreg_loss:    Optional[SIGReg] = None,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
    total_loss = 0.0
    n_batches  = 0
    last_anc = last_pos = last_neg = last_mask = None

    for batch in tqdm(dataloader_val, desc="  Eval", leave=False):
        with torch.no_grad():
            anc_out, pos_out, neg_out, neg_mask = _forward_batch(
                model, batch, args.device, args.cached
            )
            loss = _compute_loss(anc_out, pos_out, neg_out, neg_mask,
                                 nce_loss, sigreg_loss, args)

        total_loss += loss.item()
        n_batches  += 1
        last_anc, last_pos, last_neg, last_mask = anc_out, pos_out, neg_out, neg_mask

    return total_loss / max(n_batches, 1), last_anc, last_pos, last_neg, last_mask


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model:              PredictionHead,
    dataset_train:      ChampionSimilarityDataset,
    dataset_validation: ChampionSimilarityDataset,
    args:               ContrastiveTrainingArgs,
) -> None:
    mode_label = "CACHED" if args.cached else "ONLINE"
    print(f"[train] Mode: {mode_label}")

    nce_loss    = NCELoss(temperature=args.temperature)
    sigreg_loss = SIGReg(n_projections=32, lam=0.1) if args.use_sigreg else None

    dataloader_train = build_dataloader(dataset_train, batch_size=args.train_batch_size, shuffle=True)
    dataloader_val   = build_dataloader(dataset_validation, batch_size=args.eval_batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(
        model.head.parameters(),   # only the MLP head is trainable in both modes
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    logger          = MetricLogger(args.logging_path)
    best_val_loss   = float("inf")
    steps_per_epoch = len(dataloader_train)
    last_cost       = torch.tensor(float("nan"))

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        for step, batch in tqdm(
            enumerate(dataloader_train),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        ):
            global_step = step + epoch * steps_per_epoch

            anc_out, pos_out, neg_out, neg_mask = _forward_batch(
                model, batch, args.device, args.cached
            )
            cost = _compute_loss(anc_out, pos_out, neg_out, neg_mask,
                                 nce_loss, sigreg_loss, args)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            last_cost = cost

            if step % args.logging_steps == 0:
                m = logger.log_train(
                    global_step, epoch, cost.item(),
                    anc_out.detach(), pos_out.detach(), neg_out.detach(), neg_mask,
                )
                tqdm.write(
                    f"  [Train] epoch={epoch}  step={global_step}  "
                    f"loss={cost.item():.4f}  cos_pos={m['cos_pos']:.4f}  "
                    f"gap={m['gap_mean']:.4f}  erank={m['eff_rank']:.1f}"
                )

            if step % args.eval_steps == 0:
                model.eval()
                tqdm.write("  Evaluating on validation set...")
                val_loss, v_anc, v_pos, v_neg, v_mask = _eval_pass(
                    dataloader_val, model, nce_loss, args, sigreg_loss
                )
                m = logger.log_val(global_step, epoch, val_loss, v_anc, v_pos, v_neg, v_mask)
                tqdm.write(
                    f"  [Val]   epoch={epoch}  step={global_step}  "
                    f"loss={val_loss:.4f}  cos_pos={m['cos_pos']:.4f}  "
                    f"gap={m['gap_mean']:.4f}  erank={m['eff_rank']:.1f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt = os.path.join(args.output_dir,
                                        f"best_epoch{epoch+1}_step{global_step}_{val_loss:.4f}.pth")
                    torch.save(model.state_dict(), ckpt)
                    tqdm.write(f"  + New best -> {ckpt}")
                model.train()

        # -- End-of-epoch validation ----------------------------------------
        model.eval()
        val_loss, v_anc, v_pos, v_neg, v_mask = _eval_pass(
            dataloader_val, model, nce_loss, args, sigreg_loss
        )
        global_step = (epoch + 1) * steps_per_epoch
        m = logger.log_val(global_step, epoch, val_loss, v_anc, v_pos, v_neg, v_mask)

        print(f"Epoch {epoch+1}/{args.epochs}  "
              f"train_loss={last_cost.item():.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(args.output_dir,
                                f"best_epoch{epoch+1}_final_{val_loss:.4f}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"  + New best -> {ckpt}")

        model.train()

    plot_from_jsonl(args.logging_path, os.path.join(args.output_dir, "plots"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: ContrastiveTrainingArgs) -> None:
    """
    Convenience wrapper: load datasets, build model, call train().

    CACHED mode (recommended)
    -------------------------
        run(ContrastiveTrainingArgs(
            cache_path    = "cache/backbone_cache.pt",
            train_batch_size = 64,
        ))

    ONLINE mode (no cache)
    ----------------------
        run(ContrastiveTrainingArgs(
            cache_path    = None,
            backbone_name = "BAAI/bge-m3",
            train_batch_size = 16,   # lower — backbone runs every step
        ))
    """
    mode_label = "CACHED" if args.cached else f"ONLINE ({args.backbone_name})"
    print(f"[run] Mode: {mode_label}")

    dataset_train = ChampionSimilarityDataset(
        split      = "train",
        cache_dir  = args.hf_cache_dir,
        cache_path = args.cache_path,          # None → TEXT mode in DataLoader
    )
    dataset_val = ChampionSimilarityDataset(
        split      = "validation",
        cache_dir  = args.hf_cache_dir,
        cache_path = args.cache_path,
    )
    print(f"[run] Train: {len(dataset_train)} samples  |  Val: {len(dataset_val)} samples")

    # backbone_name=None → CACHED mode (no ST loaded)
    # backbone_name="BAAI/bge-m3" → ONLINE mode (ST loaded and frozen)
    model = load_model(
        output_dim     = args.output_dim,
        hidden_dim     = args.hidden_dim,
        bottleneck_dim = args.bottleneck_dim,
        dropout        = args.dropout,
        backbone_name  = None if args.cached else args.backbone_name,
        device         = args.device,
    ).to(args.device)
    model.train()

    train(model, dataset_train, dataset_val, args)
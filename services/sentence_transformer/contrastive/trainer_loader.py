from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
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
    output_dir:   str = "output/c2cp_bge"
    logging_path: str = "logs/c2cp_bge/c2cp_bge_metrics.jsonl"

    # Training hyper-parameters
    epochs:           int   = 10
    train_batch_size: int   = 32
    eval_batch_size:  int   = 64
    learning_rate:    float = 3e-4
    weight_decay:     float = 1e-2

    # Loss hyper-parameters
    temperature:  float = 0.07   # NCE temperature
    margin:       float = 0.5    # triplet margin
    p:            int   = 2      # Lp distance for triplet
    use_sigreg:   bool  = True   # add SIGReg uniformity term

    # Logging / checkpointing
    logging_steps: int = 20   # train log frequency (steps)
    eval_steps:    int = 100  # validation frequency (steps)

    # Model architecture
    output_dim:     int   = 256
    hidden_dim:     int   = 512
    bottleneck_dim: int   = 256
    dropout:        float = 0.1
    freeze_backbone: bool = True

    # HuggingFace dataset cache
    hf_cache_dir: Optional[str] = None

    # Device
    device: str = "cuda"


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class NCELoss(torch.nn.Module):
    """
    InfoNCE (NT-Xent) contrastive loss.

    All negatives in the padded block are used; padding positions are masked
    out via neg_mask so they do not contribute to the partition function.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        anchor:   torch.Tensor,   # (B, D)  L2-normalised
        positive: torch.Tensor,   # (B, D)
        negative: torch.Tensor,   # (B, K, D)
        neg_mask: torch.BoolTensor,  # (B, K)
    ) -> torch.Tensor:
        B, K, D = negative.shape

        # Positive logit
        pos_sim = (anchor * positive).sum(dim=1, keepdim=True) / self.temperature  # (B, 1)

        # Negative logits — mask padding to -inf so they don't enter softmax
        neg_sim = torch.einsum("bd,bkd->bk", anchor, negative) / self.temperature  # (B, K)
        neg_sim = neg_sim.masked_fill(~neg_mask, float("-inf"))

        # log-softmax over [pos | negs]
        logits  = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+K)
        labels  = torch.zeros(B, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)


class SIGReg(torch.nn.Module):
    """
    Random-projection uniformity regulariser (Wang & Isola, 2020 variant).
    Encourages embeddings to be uniformly distributed on the unit hypersphere
    by minimising the log-mean-exp of pairwise distances in a random subspace.

    Parameters
    ----------
    n_projections : number of random projection directions (default 32)
    lam           : regularisation weight
    """

    def __init__(self, n_projections: int = 32, lam: float = 0.05) -> None:
        super().__init__()
        self.n_projections = n_projections
        self.lam           = lam

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """embeddings: (B, D)  — already L2-normalised."""
        B, D = embeddings.shape
        W = F.normalize(
            torch.randn(D, self.n_projections, device=embeddings.device), dim=0
        )                                                   # (D, P)
        proj = embeddings @ W                               # (B, P)
        # pairwise squared L2 in the projected space
        diff = proj.unsqueeze(0) - proj.unsqueeze(1)        # (B, B, P)
        sq   = (diff ** 2).sum(dim=2)                       # (B, B)
        return self.lam * torch.log(torch.exp(-2.0 * sq).mean())


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

def _effective_rank(embeddings: torch.Tensor) -> float:
    """
    Effective rank via the entropy of the normalised singular-value spectrum.
    Roy & Vetterli (2007): erank = exp(H(p)) where p_i = sigma_i / sum(sigma).
    """
    with torch.no_grad():
        e = embeddings.float()
        _, s, _ = torch.linalg.svd(e, full_matrices=False)
        s = s[s > 0]
        p = s / s.sum()
        return float(torch.exp(-(p * torch.log(p)).sum()).item())


def _mean_pairwise_cos(embeddings: torch.Tensor) -> float:
    """Mean of the upper-triangle of the pairwise cosine-similarity matrix."""
    with torch.no_grad():
        e = F.normalize(embeddings.float(), dim=1)
        sim = e @ e.T
        N   = sim.size(0)
        if N < 2:
            return float(sim.mean().item())
        idx = torch.triu_indices(N, N, offset=1)
        return float(sim[idx[0], idx[1]].mean().item())


def _compute_metrics(
    anc:      torch.Tensor,   # (B, D)
    pos:      torch.Tensor,   # (B, D)
    neg:      torch.Tensor,   # (B, K, D)
    neg_mask: torch.BoolTensor,  # (B, K)
) -> dict:
    """Return a flat dict with all embedding-space metrics for one batch."""
    with torch.no_grad():
        anc_n = F.normalize(anc.float(), dim=1)
        pos_n = F.normalize(pos.float(), dim=1)

        cp    = (anc_n * pos_n).sum(dim=1)                        # (B,)

        neg_n = F.normalize(neg.float(), dim=2)                   # (B, K, D)
        cn    = torch.einsum("bd,bkd->bk", anc_n, neg_n)         # (B, K)
        valid = cn[neg_mask]                                       # (M,)

        cp_exp = cp.unsqueeze(1).expand_as(cn)                    # (B, K)
        gaps   = (cp_exp - cn)[neg_mask]                          # (M,)

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
# Streaming JSONL logger
# ---------------------------------------------------------------------------

class MetricLogger:
    """
    Writes one JSON line per event immediately to disk.
    The JSONL file is the single source of truth.
    Truncated at construction so each run starts clean.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, "w") as _:
            pass  # truncate / create

    def _write(self, row: dict) -> None:
        with open(self.path, "a") as f:
            json.dump(row, f)
            f.write("\n")

    def log_train(
        self,
        global_step: int,
        epoch: int,
        loss: float,
        anc: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        neg_mask: torch.BoolTensor,
    ) -> dict:
        metrics = _compute_metrics(anc, pos, neg, neg_mask)
        self._write({"type": "train", "global_step": global_step,
                     "epoch": epoch, "loss": loss, **metrics})
        return metrics

    def log_val(
        self,
        global_step: int,
        epoch: int,
        loss: float,
        anc: torch.Tensor,
        pos: torch.Tensor,
        neg: torch.Tensor,
        neg_mask: torch.BoolTensor,
    ) -> dict:
        metrics = _compute_metrics(anc, pos, neg, neg_mask)
        self._write({"type": "val", "global_step": global_step,
                     "epoch": epoch, "loss": loss, **metrics})
        return metrics


# ---------------------------------------------------------------------------
# JSONL -> dict loader
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> tuple[dict, dict]:
    """
    Parse a metrics JSONL file produced by MetricLogger.

    Returns
    -------
    train : dict[str, list]   rows where type == "train"
    val   : dict[str, list]   rows where type == "val"
    """
    train: dict[str, list] = {}
    val:   dict[str, list] = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row  = json.loads(line)
            dest = train if row["type"] == "train" else val
            for k, v in row.items():
                if k == "type":
                    continue
                dest.setdefault(k, []).append(v)

    return train, val


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_STYLE = {
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "lines.linewidth":   1.6,
}

# Wong colorblind-safe palette
_TRAIN = "#0072B2"   # blue
_VAL   = "#D55E00"   # vermillion
_NEG   = "#E69F00"   # orange


def _new_fig(title: str):
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.set_title(title, pad=8)
    return fig, ax


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _band(ax, steps, mean_vals, std_vals, color, alpha=0.16, **kw):
    """Plot a line with a ±1 std shaded band."""
    m = np.asarray(mean_vals)
    s = np.asarray(std_vals)
    ax.plot(steps, m, color=color, **kw)
    ax.fill_between(steps, m - s, m + s, color=color, alpha=alpha)


def plot_all(train: dict, val: dict, output_dir: str) -> None:
    """
    Generate all publication-ready figures from pre-loaded metric dicts.

    Parameters
    ----------
    train, val  : output of load_jsonl()
    output_dir  : directory where PNGs are written
    """
    os.makedirs(output_dir, exist_ok=True)
    has_val = bool(val.get("global_step"))

    with plt.rc_context(_STYLE):

        # 1 -- Loss ---------------------------------------------------------
        fig, ax = _new_fig("Training & Validation Loss")
        ax.plot(train["global_step"], train["loss"], color=_TRAIN, label="Train")
        if has_val:
            ax.plot(val["global_step"], val["loss"], color=_VAL,
                    linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Loss"); ax.legend()
        _save(fig, os.path.join(output_dir, "01_loss.png"))

        # 2 -- Anchor-positive cosine similarity ----------------------------
        fig, ax = _new_fig("Anchor-Positive Cosine Similarity")
        ax.plot(train["global_step"], train["cos_pos"], color=_TRAIN, label="Train")
        if has_val:
            ax.plot(val["global_step"], val["cos_pos"], color=_VAL,
                    linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Cosine similarity"); ax.legend()
        _save(fig, os.path.join(output_dir, "02_cos_pos.png"))

        # 3 -- Anchor-negative cosine similarity (mean ± std) ---------------
        fig, ax = _new_fig("Anchor-Negative Cosine Similarity  (mean ± 1 std)")
        _band(ax, train["global_step"], train["cos_neg_mean"], train["cos_neg_std"],
              _NEG, label="Train mean")
        if has_val:
            _band(ax, val["global_step"], val["cos_neg_mean"], val["cos_neg_std"],
                  _VAL, linestyle="--", label="Val mean")
        ax.set_xlabel("Global step"); ax.set_ylabel("Cosine similarity"); ax.legend()
        _save(fig, os.path.join(output_dir, "03_cos_neg.png"))

        # 4 -- Similarity gap (mean ± std) ----------------------------------
        fig, ax = _new_fig("Similarity Gap  cos(a,p) − cos(a,n)  (mean ± 1 std)")
        _band(ax, train["global_step"], train["gap_mean"], train["gap_std"],
              _TRAIN, label="Train mean")
        if has_val:
            _band(ax, val["global_step"], val["gap_mean"], val["gap_std"],
                  _VAL, linestyle="--", label="Val mean")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Global step"); ax.set_ylabel("Similarity gap"); ax.legend()
        _save(fig, os.path.join(output_dir, "04_gap.png"))

        # 5 -- Effective rank -----------------------------------------------
        fig, ax = _new_fig("Embedding Effective Rank  (Roy & Vetterli, 2007)")
        ax.plot(train["global_step"], train["eff_rank"], color=_TRAIN, label="Train")
        if has_val:
            ax.plot(val["global_step"], val["eff_rank"], color=_VAL,
                    linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Effective rank"); ax.legend()
        _save(fig, os.path.join(output_dir, "05_effective_rank.png"))

        # 6 -- Per-dimension std --------------------------------------------
        fig, ax = _new_fig("Mean Per-Dimension Std of Anchor Embeddings")
        ax.plot(train["global_step"], train["per_dim_std"], color=_TRAIN, label="Train")
        if has_val:
            ax.plot(val["global_step"], val["per_dim_std"], color=_VAL,
                    linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Mean per-dim std"); ax.legend()
        _save(fig, os.path.join(output_dir, "06_per_dim_std.png"))

        # 7 -- Mean pairwise cosine similarity ------------------------------
        fig, ax = _new_fig("Global Mean Pairwise Cosine Similarity  (batch anchors)")
        ax.plot(train["global_step"], train["mpcs"], color=_TRAIN, label="Train")
        if has_val:
            ax.plot(val["global_step"], val["mpcs"], color=_VAL,
                    linestyle="--", label="Val")
        ax.set_xlabel("Global step"); ax.set_ylabel("Mean pairwise cosine sim"); ax.legend()
        _save(fig, os.path.join(output_dir, "07_mean_pairwise_cos.png"))

        # 8 -- Collapse overview (3-panel) ----------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
        panels = [
            ("Effective Rank",    "eff_rank"),
            ("Per-Dim Std",       "per_dim_std"),
            ("Mean Pairwise Cos", "mpcs"),
        ]
        for ax, (title, key) in zip(axes, panels):
            ax.set_title(title, fontsize=11)
            ax.plot(train["global_step"], train[key], color=_TRAIN, label="Train")
            if has_val:
                ax.plot(val["global_step"], val[key], color=_VAL,
                        linestyle="--", label="Val")
            ax.set_xlabel("Step")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(alpha=0.35)
            ax.legend(fontsize=8)
        fig.suptitle("Representation Collapse Diagnostics", fontsize=12, y=1.01)
        _save(fig, os.path.join(output_dir, "08_collapse_overview.png"))

    print(f"[Plotter] 8 figures saved to {output_dir}")


def plot_from_jsonl(jsonl_path: str, output_dir: str) -> None:
    """
    Standalone entry point — re-generate all figures from a saved JSONL file.

    Usage
    -----
        from trainer_loader import plot_from_jsonl
        plot_from_jsonl("logs/metrics.jsonl", "plots/run_01/")
    """
    train, val = load_jsonl(jsonl_path)
    plot_all(train, val, output_dir)


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
    """
    Full validation pass.

    Returns
    -------
    (mean_loss, last_anc, last_pos, last_neg, last_neg_mask)
    The last-batch embeddings are used as a representative sample for logging.
    """
    total_loss = 0.0
    n_batches  = 0
    last_anc = last_pos = last_neg = last_mask = None

    for anchor_texts, positive_texts, negative_texts, neg_mask in tqdm(
        dataloader_val, desc="  Eval", leave=False
    ):
        neg_mask = neg_mask.to(args.device)

        with torch.no_grad():
            anc_emb           = model(anchor_texts)
            pos_emb           = model(positive_texts)
            neg_emb, neg_mask = model.encode_negatives(negative_texts)

            # Triplet margin loss over valid negatives
            K       = neg_emb.size(1)
            anc_rep = anc_emb.unsqueeze(1).expand(-1, K, -1)
            pos_rep = pos_emb.unsqueeze(1).expand(-1, K, -1)

            triplet = F.relu(
                F.pairwise_distance(anc_rep[neg_mask], pos_rep[neg_mask], p=args.p)
                - F.pairwise_distance(anc_rep[neg_mask], neg_emb[neg_mask], p=args.p)
                + args.margin
            ).mean()

            nce = nce_loss(anc_emb, pos_emb, neg_emb, neg_mask)

            if args.use_sigreg and sigreg_loss is not None:
                loss = nce + 0.5 * triplet + 0.5 * (
                    sigreg_loss(anc_emb) + sigreg_loss(pos_emb)
                )
            else:
                loss = nce + 0.5 * triplet

        total_loss += loss.item()
        n_batches  += 1
        last_anc, last_pos, last_neg, last_mask = anc_emb, pos_emb, neg_emb, neg_mask

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
    """
    Main contrastive training loop.

    Parameters
    ----------
    model              : PredictionHead (already on args.device)
    dataset_train      : training split of ChampionSimilarityDataset
    dataset_validation : validation split
    args               : ContrastiveTrainingArgs
    """
    nce_loss    = NCELoss(temperature=args.temperature)
    sigreg_loss = SIGReg(n_projections=32, lam=0.05) if args.use_sigreg else None

    dataloader_train = build_dataloader(
        dataset_train,
        batch_size=args.train_batch_size,
        shuffle=True,
    )
    dataloader_val = build_dataloader(
        dataset_validation,
        batch_size=args.eval_batch_size,
        shuffle=False,
    )

    # Only the head is trainable — backbone is frozen
    optimizer = torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": args.learning_rate}],
        weight_decay=args.weight_decay,
    )

    logger          = MetricLogger(args.logging_path)
    best_val_loss   = float("inf")
    steps_per_epoch = len(dataloader_train)
    last_cost       = torch.tensor(float("nan"))

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        for step, (anchor_texts, positive_texts, negative_texts, neg_mask) in tqdm(
            enumerate(dataloader_train),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        ):
            neg_mask   = neg_mask.to(args.device)
            global_step = step + epoch * steps_per_epoch

            # -- Forward ----------------------------------------------------
            anc_emb           = model(anchor_texts)
            pos_emb           = model(positive_texts)
            neg_emb, neg_mask = model.encode_negatives(negative_texts)

            K       = neg_emb.size(1)
            anc_rep = anc_emb.unsqueeze(1).expand(-1, K, -1)
            pos_rep = pos_emb.unsqueeze(1).expand(-1, K, -1)

            triplet = F.relu(
                F.pairwise_distance(anc_rep[neg_mask], pos_rep[neg_mask], p=args.p)
                - F.pairwise_distance(anc_rep[neg_mask], neg_emb[neg_mask], p=args.p)
                + args.margin
            ).mean()

            nce = nce_loss(anc_emb, pos_emb, neg_emb, neg_mask)

            if args.use_sigreg and sigreg_loss is not None:
                cost = nce + 0.5 * triplet + 0.5 * (
                    sigreg_loss(anc_emb) + sigreg_loss(pos_emb)
                )
            else:
                cost = nce + 0.5 * triplet

            # -- Backward ---------------------------------------------------
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            last_cost = cost

            # -- Train logging ----------------------------------------------
            if step % args.logging_steps == 0:
                m = logger.log_train(
                    global_step, epoch, cost.item(),
                    anc_emb.detach(), pos_emb.detach(),
                    neg_emb.detach(), neg_mask,
                )
                tqdm.write(
                    f"  [Train] epoch={epoch}  step={global_step}  "
                    f"loss={cost.item():.4f}  cos_pos={m['cos_pos']:.4f}  "
                    f"gap={m['gap_mean']:.4f}  erank={m['eff_rank']:.1f}"
                )

            # -- Mid-epoch validation ---------------------------------------
            if step % args.eval_steps == 0:
                model.eval()
                tqdm.write("  Evaluating on validation set...")
                val_loss, v_anc, v_pos, v_neg, v_mask = _eval_pass(
                    dataloader_val, model, nce_loss, args, sigreg_loss
                )
                m = logger.log_val(
                    global_step, epoch, val_loss,
                    v_anc, v_pos, v_neg, v_mask,
                )
                tqdm.write(
                    f"  [Val]   epoch={epoch}  step={global_step}  "
                    f"loss={val_loss:.4f}  cos_pos={m['cos_pos']:.4f}  "
                    f"gap={m['gap_mean']:.4f}  erank={m['eff_rank']:.1f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    ckpt = os.path.join(
                        args.output_dir,
                        f"best_epoch{epoch+1}_step{global_step}_{val_loss:.4f}.pth",
                    )
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

        print(
            f"Epoch {epoch+1}/{args.epochs}  "
            f"train_loss={last_cost.item():.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(
                args.output_dir,
                f"best_epoch{epoch+1}_final_{val_loss:.4f}.pth",
            )
            torch.save(model.state_dict(), ckpt)
            print(f"  + New best -> {ckpt}")

        model.train()

    # -- Final plots --------------------------------------------------------
    plots_dir = os.path.join(args.output_dir, "plots")
    plot_from_jsonl(args.logging_path, plots_dir)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(args: ContrastiveTrainingArgs) -> None:
    """
    Convenience wrapper: load datasets, build model, call train().

    Usage
    -----
        from trainer_loader import run, ContrastiveTrainingArgs
        run(ContrastiveTrainingArgs(device="cuda", epochs=20))
    """
    print(f"[run] Loading datasets from HuggingFace (cache: {args.hf_cache_dir})")
    dataset_train = ChampionSimilarityDataset(split="train",      cache_dir=args.hf_cache_dir)
    dataset_val   = ChampionSimilarityDataset(split="validation", cache_dir=args.hf_cache_dir)

    print(f"[run] Train: {len(dataset_train)} samples  |  Val: {len(dataset_val)} samples")

    print(f"[run] Building model (output_dim={args.output_dim}, freeze_backbone={args.freeze_backbone})")
    model = load_model(
        output_dim      = args.output_dim,
        hidden_dim      = args.hidden_dim,
        bottleneck_dim  = args.bottleneck_dim,
        dropout         = args.dropout,
        freeze_backbone = args.freeze_backbone,
        device          = args.device,
    ).to(args.device)

    # Switch head to train mode explicitly (backbone stays eval when frozen)
    model.head.train()

    print("[run] Starting training...")
    train(model, dataset_train, dataset_val, args)
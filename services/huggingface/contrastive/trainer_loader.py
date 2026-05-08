"""
trainer_loader.py
-----------------
Contrastive trainer for PredictionHead with rich metric tracking and
publication-ready figures.

Tracked metrics (logged at every logging_steps / eval_steps):
  Training rows  (type == "train"):
    global_step, epoch, loss,
    cos_pos,
    cos_neg_mean, cos_neg_std,
    gap_mean, gap_std,
    eff_rank, per_dim_std, mpcs

  Validation rows  (type == "val"):
    global_step, epoch, loss,
    cos_pos,
    cos_neg_mean, cos_neg_std,
    gap_mean, gap_std,
    eff_rank, per_dim_std, mpcs

Every row is written to the JSONL log file immediately after it is computed,
so figures can be re-generated at any time — even mid-training — via:

    from trainer_loader import plot_from_jsonl
    plot_from_jsonl("logs/metrics.jsonl", "plots/")
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from services.huggingface.contrastive.loss import NCELoss, SIGReg
from utils.types.TrainingArgs import ContrastiveTrainingArgs


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
        N = sim.size(0)
        if N < 2:
            return float(sim.mean().item())
        idx = torch.triu_indices(N, N, offset=1)
        return float(sim[idx[0], idx[1]].mean().item())


def _compute_metrics(
    anc: torch.Tensor,        # (B, D)
    pos: torch.Tensor,        # (B, D)
    neg: torch.Tensor,        # (B, K, D)  zero-padded
    neg_mask: torch.Tensor,   # (B, K) bool
) -> dict:
    """
    Return a flat dict with all embedding-space metrics for one batch.
    Shared between train and val logging paths.
    """
    with torch.no_grad():
        anc_n = F.normalize(anc.float(), dim=1)
        pos_n = F.normalize(pos.float(), dim=1)

        # Anchor-positive cosine similarity
        cp = (anc_n * pos_n).sum(dim=1)                        # (B,)

        # Anchor-negative cosine similarity (valid negatives only)
        neg_n = F.normalize(neg.float(), dim=2)                # (B, K, D)
        cn    = torch.einsum("bd,bkd->bk", anc_n, neg_n)      # (B, K)
        valid = cn[neg_mask]                                    # (M,)

        # Gap: cos_pos[b] - cos_neg[b, k] for every valid pair
        cp_exp = cp.unsqueeze(1).expand_as(cn)                 # (B, K)
        gaps   = (cp_exp - cn)[neg_mask]                       # (M,)

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
    Keeps no in-memory history — the JSONL file is the single source of truth.
    The file is truncated at construction so each training run starts clean.
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
        neg_mask: torch.Tensor,
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
        neg_mask: torch.Tensor,
    ) -> dict:
        metrics = _compute_metrics(anc, pos, neg, neg_mask)
        self._write({"type": "val", "global_step": global_step,
                     "epoch": epoch, "loss": loss, **metrics})
        return metrics


# ---------------------------------------------------------------------------
# JSONL -> dict loader  (used by the plotting layer)
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
# Plotting  (works purely from plain dicts — no PyTorch dependency)
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

_TRAIN = "#2563EB"   # blue
_VAL   = "#DC2626"   # red
_NEG   = "#F97316"   # orange


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

        # 3 -- Anchor-negative cosine similarity (mean +/- std) -------------
        fig, ax = _new_fig("Anchor-Negative Cosine Similarity  (mean +/- 1 std)")
        _band(ax, train["global_step"], train["cos_neg_mean"], train["cos_neg_std"],
              _NEG, label="Train mean")
        if has_val:
            _band(ax, val["global_step"], val["cos_neg_mean"], val["cos_neg_std"],
                  _VAL, linestyle="--", label="Val mean")
        ax.set_xlabel("Global step"); ax.set_ylabel("Cosine similarity"); ax.legend()
        _save(fig, os.path.join(output_dir, "03_cos_neg.png"))

        # 4 -- Similarity gap (mean +/- std) --------------------------------
        fig, ax = _new_fig("Similarity Gap  cos(a,p) - cos(a,n)  (mean +/- 1 std)")
        _band(ax, train["global_step"], train["gap_mean"], train["gap_std"],
              _TRAIN, label="Train mean")
        if has_val:
            _band(ax, val["global_step"], val["gap_mean"], val["gap_std"],
                  _VAL, linestyle="--", label="Val mean")
        ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
        ax.set_xlabel("Global step"); ax.set_ylabel("Similarity gap"); ax.legend()
        _save(fig, os.path.join(output_dir, "04_gap.png"))

        # 5 -- Embedding effective rank -------------------------------------
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
    No model or training state required.

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
    model: torch.nn.Module,
    nce_loss: NCELoss,
    training_args: ContrastiveTrainingArgs,
    sigreg_loss: SIGReg = None,
) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full validation pass.
    Returns (mean_loss, last_anc, last_pos, last_neg_packed, last_neg_mask).
    The last-batch embeddings are used as a representative sample for logging.
    """
    assert training_args.use_sigreg and sigreg_loss is not None
    total_loss = 0.0
    n_batches  = 0
    last_anc = last_pos = last_neg = last_mask = None

    for anchor_val, positive_val, negative_val in tqdm(
        dataloader_val, desc="  Eval", leave=False
    ):
        with torch.no_grad():
            anc_emb = model(anchor_val)
            pos_emb = model(positive_val)
            neg_emb, neg_mask = model.encode_negatives(negative_val)

            K       = neg_emb.size(1)
            anc_rep = anc_emb.unsqueeze(1).expand(-1, K, -1)
            pos_rep = pos_emb.unsqueeze(1).expand(-1, K, -1)

            triplet = F.relu(
                F.pairwise_distance(anc_rep[neg_mask], pos_rep[neg_mask],
                                    p=training_args.p) -
                F.pairwise_distance(anc_rep[neg_mask], neg_emb[neg_mask],
                                    p=training_args.p) +
                training_args.margin
            ).mean()
            if training_args.use_sigreg:
                loss = nce_loss(anc_emb, pos_emb, neg_emb) + 0.5 * triplet + 0.5 * (sigreg_loss(anc_emb) + sigreg_loss(pos_emb))
            else:
                loss = nce_loss(anc_emb, pos_emb, neg_emb) + 0.5 * triplet

        total_loss += loss.item()
        n_batches  += 1
        last_anc, last_pos, last_neg, last_mask = anc_emb, pos_emb, neg_emb, neg_mask

    return total_loss / max(n_batches, 1), last_anc, last_pos, last_neg, last_mask


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset_train: torch.utils.data.Dataset,
    dataset_validation: torch.utils.data.Dataset,
    training_args: ContrastiveTrainingArgs,
) -> None:
    from services.huggingface.contrastive.data_loader import build_dataloader

    nce_loss = NCELoss(temperature=training_args.temperature)
    sigreg_loss = SIGReg(n_projections=32, lam=0.05)

    dataloader_train = build_dataloader(
        dataset_train, batch_size=training_args.train_batch_size, shuffle=True
    )
    dataloader_val = build_dataloader(
        dataset_validation, batch_size=training_args.eval_batch_size, shuffle=False
    )

    logger          = MetricLogger(training_args.logging_path)
    best_val_loss   = float("inf")
    steps_per_epoch = len(dataloader_train)

    os.makedirs(training_args.output_dir, exist_ok=True)

    for epoch in range(training_args.epochs):
        model.train()

        for step, (anchor_batch, positive_batch, negative_batch) in tqdm(
            enumerate(dataloader_train),
            total=steps_per_epoch,
            desc=f"Epoch {epoch + 1}/{training_args.epochs}",
        ):
            global_step = step + epoch * steps_per_epoch

            # -- Forward ----------------------------------------------------
            anc_emb = model(anchor_batch)
            pos_emb = model(positive_batch)
            neg_emb, neg_mask = model.encode_negatives(negative_batch)

            K       = neg_emb.size(1)
            anc_rep = anc_emb.unsqueeze(1).expand(-1, K, -1)
            pos_rep = pos_emb.unsqueeze(1).expand(-1, K, -1)

            triplet = F.relu(
                F.pairwise_distance(anc_rep[neg_mask], pos_rep[neg_mask],
                                    p=training_args.p) -
                F.pairwise_distance(anc_rep[neg_mask], neg_emb[neg_mask],
                                    p=training_args.p) +
                training_args.margin
            ).mean()
            if training_args.use_sigreg:
                cost = nce_loss(anc_emb, pos_emb, neg_emb) + 0.5 * triplet + 1/2 * (sigreg_loss(anc_emb) + sigreg_loss(pos_emb))
            else:
                cost = nce_loss(anc_emb, pos_emb, neg_emb) + 0.5 * triplet

            # -- Backward ---------------------------------------------------
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # -- Train logging ----------------------------------------------
            if step % training_args.logging_steps == 0:
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

            # -- Validation -------------------------------------------------
            if step % training_args.eval_steps == 0:
                model.eval()
                tqdm.write("  Evaluating on validation set...")
                if training_args.use_sigreg:
                    val_loss, v_anc, v_pos, v_neg, v_mask = _eval_pass(
                        dataloader_val, model, nce_loss, training_args, sigreg_loss
                    )
                else:
                    val_loss, v_anc, v_pos, v_neg, v_mask = _eval_pass(
                        dataloader_val, model, nce_loss, training_args
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
                        training_args.output_dir,
                        f"best_epoch{epoch+1}_step{global_step}_{val_loss:.4f}.pth",
                    )
                    torch.save(model.state_dict(), ckpt)
                    tqdm.write(f"  + New best -> {ckpt}")

                model.train()

        # -- End-of-epoch validation ----------------------------------------
        model.eval()
        if training_args.use_sigreg:
            val_loss, v_anc, v_pos, v_neg, v_mask = _eval_pass(
                dataloader_val, model, nce_loss, training_args, sigreg_loss
            )
        else:
            val_loss, v_anc, v_pos, v_neg, v_mask = _eval_pass(
                dataloader_val, model, nce_loss, training_args
            )
        global_step = (epoch + 1) * steps_per_epoch
        m = logger.log_val(global_step, epoch, val_loss, v_anc, v_pos, v_neg, v_mask)

        print(
            f"Epoch {epoch+1}/{training_args.epochs}  "
            f"train_loss={cost.item():.4f}  val_loss={val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = os.path.join(
                training_args.output_dir,
                f"best_epoch{epoch+1}_final_{val_loss:.4f}.pth",
            )
            torch.save(model.state_dict(), ckpt)
            print(f"  + New best -> {ckpt}")

        model.train()

    # -- Final plots --------------------------------------------------------
    plots_dir = os.path.join(training_args.output_dir, "plots")
    plot_from_jsonl(training_args.logging_path, plots_dir)
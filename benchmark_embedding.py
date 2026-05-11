"""
benchmark_c2cp.py
=================
Benchmark script for the C2CP champion embedding model.

Evaluates whether the embedding model correctly captures cross-role
mechanical playstyle similarity rather than role-archetype clustering.

Usage
-----
    python benchmark_c2cp.py \
        --rationales  ./data/champion_rationales.jsonl \
        --embeddings  ./data/champions_embeddings.json \
        --out         ./output/benchmark_c2cp/

The script expects either:
  - --embeddings: a pre-computed JSON dict {champion_name: [float, ...]}
  - --model + --rationales: a model path to compute embeddings on the fly
    (requires your PredictionHead class to be importable)

All figures are saved as publication-ready PDFs + PNGs (300 Dpi, serif font).
"""

import argparse
import json
import math
import os
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.spatial.distance import cosine
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "mathtext.fontset":   "dejavuserif",
    "axes.titlesize":     8,
    "axes.titleweight":   "bold",
    "axes.labelsize":     7,
    "xtick.labelsize":    6,
    "ytick.labelsize":    6,
    "legend.fontsize":    6,
    "legend.framealpha":  0.92,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth":     0.6,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.5,
    "lines.linewidth":    1.1,
})

# ── Ground-truth mechanical similarity groups ──────────────────────────────────
# Each group contains champions that share an execution pattern across roles.
# Used for qualitative nearest-neighbour evaluation.
# Expand this dict freely — it is the primary benchmark oracle.
MECHANICAL_GROUPS = {
    "single_commit_teamfight": [
        "Malphite", "Amumu", "Galio", "Zac", "Orianna",
    ],
    "invisibility_burst_escape": [
        "Zed", "Akali", "Talon", "Kha'Zix",
    ],
    "stationary_hyperscaler": [
        "Jinx", "Kog'Maw", "Twitch",
    ],
    "zone_placement_burst_confirm": [
        "Veigar", "Lissandra", "Annie",
    ],
    "hook_single_target_lockdown": [
        "Thresh", "Blitzcrank", "Nautilus",
    ],
    "ally_synchronisation": [
        "Seraphine", "Taric", "Lulu",
    ],
    "stack_accumulation_execute": [
        "Nasus", "Veigar", "Sion",
    ],
    "mark_and_execute": [
        "Rek'Sai", "Zed", "Rengar",
    ],
    "terrain_control": [
        "Anivia", "Trundle", "Jarvan IV",
    ],
    "shadow_positional_algebra": [
        "Zed", "LeBlanc", "Ekko",
    ],
    "setup_confirm_loop": [
        "Nidalee", "Jayce", "Elise",
    ],
}

# Cross-role pairs that SHOULD be close (same execution pattern, different role)
EXPECTED_CLOSE_PAIRS = [
    ("Malphite",  "Amumu",      "single-commit teamfight initiators"),
    ("Zed",       "Akali",      "invisibility approach → burst → escape"),
    ("Jinx",      "Kog'Maw",    "stationary hyperscalers"),
    ("Veigar",    "Annie",      "zone-placement burst confirm"),
    ("Thresh",    "Nautilus",   "hook single-target lockdown"),
    ("Nidalee",   "Jayce",      "form-switch setup-confirm loop"),
    ("Rek'Sai",   "Rengar",     "mark-and-execute from stealth"),
    ("Nasus",     "Veigar",     "stack accumulation threshold spike"),
    ("LeBlanc",   "Ekko",       "shadow/temporal positional algebra"),
    ("Seraphine", "Lulu",       "ally synchronisation buffer"),
]

# Pairs that should be FAR despite sharing role label
EXPECTED_FAR_PAIRS = [
    ("Malphite",  "Darius",     "both melee but commit vs. sustained brawl"),
    ("Zed",       "Yasuo",      "both physical but shadow vs. wind wall pattern"),
    ("Jinx",      "Ezreal",     "both ranged but stationary vs. constant reposition"),
    ("Thresh",    "Soraka",     "both support but hook vs. global heal"),
    ("Nasus",     "Darius",     "both top but patience vs. aggression"),
    ("Annie",     "Syndra",     "both burst but scripted vs. sphere management"),
]


# ── Utilities ──────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - cosine(a, b))


def effective_rank(embeddings: np.ndarray) -> float:
    """Effective rank via entropy of normalised singular value spectrum."""
    try:
        _, S, _ = np.linalg.svd(embeddings.astype(np.float32), full_matrices=False)
        S = S[S > 1e-8]
        p = S / S.sum()
        return float(np.exp(-(p * np.log(p)).sum()))
    except Exception:
        return float("nan")


def mean_pairwise_cosine(embeddings: np.ndarray) -> float:
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    normed = arr / (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8)
    cos_mat = normed @ normed.T
    n = arr.shape[0]
    if n < 2:
        return float("nan")
    mask = ~np.eye(n, dtype=bool)
    return float(cos_mat[mask].mean())


def per_dim_std(embeddings: np.ndarray) -> float:
    return float(embeddings.std(axis=0).mean())


def load_embeddings(path: str) -> dict[str, np.ndarray]:
    with open(path) as f:
        raw = json.load(f)
    fixed = {}
    for name, vec in raw.items():
        arr = np.asarray(vec, dtype=np.float32)
        if arr.ndim > 1:
            arr = np.squeeze(arr)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        fixed[name] = arr
    return fixed


def load_rationales(path: str) -> dict[str, str]:
    rationales = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rationales[obj["name"]] = obj["rationale"]
    return rationales


def get_group_for_champion(champion: str) -> str | None:
    for group, members in MECHANICAL_GROUPS.items():
        if champion in members:
            return group
    return None


def fmt_ax(ax, xlabel=None, ylabel=None, title=None, grid=True):
    if title:
        ax.set_title(title, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, axis="y", ls=":", lw=0.4)
    ax.tick_params(axis="both", direction="in")


# ── Benchmark tasks ────────────────────────────────────────────────────────────

def task_global_geometry(
    embeddings: dict[str, np.ndarray],
    out_dir: Path,
) -> dict:
    """
    Task 1: Global geometry health metrics.
    Reports effective rank, mean pairwise cosine sim, and per-dim std
    over the full champion embedding set.
    """
    names  = list(embeddings.keys())
    embs   = np.stack([embeddings[n] for n in names])

    eff_r  = effective_rank(embs)
    mpc    = mean_pairwise_cosine(embs)
    std    = per_dim_std(embs)

    # PCA explained variance
    from numpy.linalg import svd
    embs_c = embs - embs.mean(axis=0)
    _, S, _ = svd(embs_c.astype(np.float32), full_matrices=False)
    var_exp = (S ** 2) / (S ** 2).sum()
    cum_var = np.cumsum(var_exp)

    # Thresholds
    dim_50  = int(np.searchsorted(cum_var, 0.50)) + 1
    dim_80  = int(np.searchsorted(cum_var, 0.80)) + 1
    dim_95  = int(np.searchsorted(cum_var, 0.95)) + 1

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    fig.suptitle("(1) Global Embedding Geometry", fontsize=9, fontweight="bold", y=1.01)

    # PCA cumulative variance
    ax = axes[0]
    x  = np.arange(1, len(cum_var) + 1)
    ax.plot(x[:200], cum_var[:200], lw=1.2, color="#1f77b4")
    ax.fill_between(x[:200], cum_var[:200], alpha=0.12, color="#1f77b4")
    for threshold, dim, clr in [
        (0.50, dim_50, "#e68a00"),
        (0.80, dim_80, "#2ca02c"),
        (0.95, dim_95, "#d62728"),
    ]:
        ax.axhline(threshold, color=clr, ls="--", lw=0.8, alpha=0.7)
        ax.axvline(dim, color=clr, ls="--", lw=0.8, alpha=0.7)
        ax.scatter([dim], [threshold], color=clr, s=18, zorder=5)
        ax.annotate(
            f"{int(threshold*100)}% → dim {dim}",
            xy=(dim, threshold), xytext=(dim + 3, threshold - 0.04),
            fontsize=5.5, color=clr,
        )
    ax.axvline(eff_r, color="gray", ls=":", lw=0.8)
    ax.text(eff_r + 1, 0.05, f"eff. rank = {eff_r:.1f}", fontsize=5.5, color="gray")
    fmt_ax(ax, xlabel="Number of PCA components",
           ylabel="Cumulative explained variance",
           title="(a) PCA Explained Variance", grid=False)
    ax.set_xlim(0, 150)
    ax.set_ylim(0, 1.05)

    # Pairwise cosine sim distribution
    ax   = axes[1]
    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    cos_mat = normed @ normed.T
    n = len(embs)
    mask = ~np.eye(n, dtype=bool)
    cos_vals = cos_mat[mask]
    ax.hist(cos_vals, bins=60, color="#1f77b4", alpha=0.8, edgecolor="none")
    ax.axvline(mpc, color="#d62728", ls="--", lw=1.0, label=f"μ={mpc:.3f}")
    ax.axvline(0.474, color="gray", ls=":", lw=0.8, label="v5 indiv. ref (0.474)")
    ax.legend(loc="upper left")
    fmt_ax(ax, xlabel="Pairwise cosine similarity",
           ylabel="Count", title="(b) Pairwise Cosine Sim Distribution")
    ax.grid(True, axis="x", ls=":", lw=0.4)

    # Per-dimension std (sorted descending)
    ax = axes[2]
    dim_stds = embs.std(axis=0)
    sorted_stds = np.sort(dim_stds)[::-1]
    ax.fill_between(range(len(sorted_stds)), sorted_stds,
                    color="#9467bd", alpha=0.7)
    ax.axhline(std, color="#d62728", ls="--", lw=1.0, label=f"μ={std:.4f}")
    ax.axhline(0.01, color="gray", ls=":", lw=0.8, label="dead-dim (0.01)")
    ax.legend()
    fmt_ax(ax, xlabel="Dimension (sorted by σ)",
           ylabel="Standard deviation",
           title="(c) Per-Dimension Std", grid=False)

    fig.tight_layout()
    fig.savefig(out_dir / "task1_global_geometry.pdf")
    fig.savefig(out_dir / "task1_global_geometry.png", dpi=300)
    plt.close(fig)

    return {
        "effective_rank":       eff_r,
        "mean_pairwise_cos":    mpc,
        "per_dim_std":          std,
        "dim_50pct_variance":   dim_50,
        "dim_80pct_variance":   dim_80,
        "dim_95pct_variance":   dim_95,
        "n_dead_dims":          int((dim_stds < 0.01).sum()),
    }


def task_expected_pairs(
    embeddings: dict[str, np.ndarray],
    out_dir:    Path,
) -> dict:
    """
    Task 2: Expected close / far pair evaluation.
    Computes cosine similarity for hand-curated mechanically similar
    and dissimilar pairs and checks that close > far.
    """
    close_sims, far_sims = [], []
    close_labels, far_labels = [], []
    missing = []

    for a, b, label in EXPECTED_CLOSE_PAIRS:
        if a not in embeddings or b not in embeddings:
            missing.append((a, b))
            continue
        sim = cosine_similarity(embeddings[a], embeddings[b])
        close_sims.append(sim)
        close_labels.append(f"{a}\n{b}")

    for a, b, label in EXPECTED_FAR_PAIRS:
        if a not in embeddings or b not in embeddings:
            missing.append((a, b))
            continue
        sim = cosine_similarity(embeddings[a], embeddings[b])
        far_sims.append(sim)
        far_labels.append(f"{a}\n{b}")

    mean_close = float(np.mean(close_sims)) if close_sims else float("nan")
    mean_far   = float(np.mean(far_sims))   if far_sims   else float("nan")
    gap        = mean_close - mean_far

    # Fraction of pairs correctly ordered (close_sim > mean_far)
    n_correct = sum(s > mean_far for s in close_sims)
    precision  = n_correct / len(close_sims) if close_sims else float("nan")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.suptitle("(2) Expected Close / Far Pair Evaluation",
                 fontsize=9, fontweight="bold", y=1.01)

    colors_close = ["#2ca02c"] * len(close_sims)
    colors_far   = ["#d62728"] * len(far_sims)

    ax = axes[0]
    y  = range(len(close_sims))
    bars = ax.barh(y, close_sims, color=colors_close, alpha=0.8, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(close_labels, fontsize=5)
    ax.axvline(mean_close, color="#2ca02c", ls="--", lw=1.0,
               label=f"mean={mean_close:.3f}")
    ax.axvline(mean_far, color="#d62728", ls="--", lw=1.0,
               label=f"far mean={mean_far:.3f}")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=5.5)
    ax.set_xlabel("Cosine similarity")
    ax.set_title("(a) Mechanically Similar Pairs (should be HIGH)", pad=4,
                 fontweight="bold", fontsize=7)
    ax.grid(True, axis="x", ls=":", lw=0.4)

    ax = axes[1]
    y  = range(len(far_sims))
    ax.barh(y, far_sims, color=colors_far, alpha=0.8, height=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(far_labels, fontsize=5)
    ax.axvline(mean_far, color="#d62728", ls="--", lw=1.0,
               label=f"mean={mean_far:.3f}")
    ax.axvline(mean_close, color="#2ca02c", ls="--", lw=1.0,
               label=f"close mean={mean_close:.3f}")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=5.5)
    ax.set_xlabel("Cosine similarity")
    ax.set_title("(b) Role-Same but Mechanically Distant Pairs (should be LOW)",
                 pad=4, fontweight="bold", fontsize=7)
    ax.grid(True, axis="x", ls=":", lw=0.4)

    fig.tight_layout()
    fig.savefig(out_dir / "task2_expected_pairs.pdf")
    fig.savefig(out_dir / "task2_expected_pairs.png", dpi=300)
    plt.close(fig)

    return {
        "mean_close_sim":        mean_close,
        "mean_far_sim":          mean_far,
        "gap_close_minus_far":   gap,
        "close_precision":       precision,
        "n_correct_close_pairs": n_correct,
        "n_total_close_pairs":   len(close_sims),
        "missing_pairs":         missing,
    }


def task_nearest_neighbours(
    embeddings: dict[str, np.ndarray],
    out_dir:    Path,
    top_k:      int = 5,
) -> dict:
    """
    Task 3: Nearest-neighbour recall within mechanical groups.
    For each champion in a known mechanical group, checks whether
    top-k nearest neighbours contain other members of the same group.
    Reports precision@k and recall@k per group.
    """
    names = list(embeddings.keys())
    embs  = np.stack([embeddings[n] for n in names])
    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    cos_mat = normed @ normed.T

    group_results = {}
    all_precisions = []

    for group, members in MECHANICAL_GROUPS.items():
        present = [m for m in members if m in embeddings]
        if len(present) < 2:
            continue

        group_prec = []
        for champion in present:
            idx   = names.index(champion)
            sims  = cos_mat[idx].copy()
            sims[idx] = -2.0  # exclude self
            top_k_idx = np.argsort(sims)[::-1][:top_k]
            top_k_names = [names[i] for i in top_k_idx]
            hits  = sum(1 for n in top_k_names if n in present and n != champion)
            prec  = hits / top_k
            group_prec.append(prec)

        mean_prec = float(np.mean(group_prec))
        group_results[group] = {
            "mean_precision_at_k": mean_prec,
            "members_found":       present,
            "k":                   top_k,
        }
        all_precisions.append(mean_prec)

    overall_precision = float(np.mean(all_precisions)) if all_precisions else float("nan")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"(3) Nearest-Neighbour Precision@{top_k} by Mechanical Group",
                 fontsize=9, fontweight="bold")

    groups     = list(group_results.keys())
    precisions = [group_results[g]["mean_precision_at_k"] for g in groups]
    colors     = ["#2ca02c" if p >= 0.5 else "#e68a00" if p >= 0.2 else "#d62728"
                  for p in precisions]

    y = range(len(groups))
    ax.barh(y, precisions, color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(y)
    group_labels = [g.replace("_", " ") for g in groups]
    ax.set_yticklabels(group_labels, fontsize=6)
    ax.axvline(overall_precision, color="navy", ls="--", lw=1.0,
               label=f"overall mean = {overall_precision:.3f}")
    ax.axvline(1/top_k, color="gray", ls=":", lw=0.8,
               label=f"random baseline = {1/top_k:.2f}")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"Precision@{top_k}")
    ax.legend(fontsize=6)
    ax.grid(True, axis="x", ls=":", lw=0.4)
    ax.tick_params(axis="both", direction="in")
    fig.tight_layout()
    fig.savefig(out_dir / "task3_nn_recall.pdf")
    fig.savefig(out_dir / "task3_nn_recall.png", dpi=300)
    plt.close(fig)

    return {
        "overall_precision_at_k": overall_precision,
        "k":                      top_k,
        "per_group":              group_results,
        "random_baseline":        1 / top_k,
    }


def task_similarity_heatmap(
    embeddings: dict[str, np.ndarray],
    out_dir:    Path,
) -> dict:
    """
    Task 4: Hierarchical clustering heatmap over curated champion subset.
    Checks whether clusters align with mechanical groups rather than roles.
    """
    # Select champions that appear in at least one mechanical group
    curated = []
    for members in MECHANICAL_GROUPS.values():
        for m in members:
            if m in embeddings and m not in curated:
                curated.append(m)

    if len(curated) < 4:
        return {"error": "too few champions in mechanical groups to cluster"}

    embs   = np.stack([embeddings[n] for n in curated])
    normed = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)
    cos_mat = normed @ normed.T
    dist_mat = 1.0 - cos_mat
    np.fill_diagonal(dist_mat, 0.0)

    # Colour-code by mechanical group
    group_colors = {
        "single_commit_teamfight":  "#1f77b4",
        "invisibility_burst_escape":"#d62728",
        "stationary_hyperscaler":  "#2ca02c",
        "zone_placement_burst_confirm": "#9467bd",
        "hook_single_target_lockdown":  "#e68a00",
        "ally_synchronisation":    "#17becf",
        "stack_accumulation_execute":   "#8c564b",
        "mark_and_execute":        "#e377c2",
        "terrain_control":         "#7f7f7f",
        "shadow_positional_algebra":    "#bcbd22",
        "setup_confirm_loop":      "#aec7e8",
    }

    def champion_color(champ):
        for group, members in MECHANICAL_GROUPS.items():
            if champ in members:
                return group_colors.get(group, "#333333")
        return "#333333"

    row_colors = [champion_color(c) for c in curated]

    # Hierarchical clustering
    linkage_mat = linkage(dist_mat[np.triu_indices(len(curated), k=1)],
                          method="average")

    fig = plt.figure(figsize=(10, 8))
    gs  = gridspec.GridSpec(2, 2, height_ratios=[1, 4], width_ratios=[4, 1],
                            hspace=0.02, wspace=0.02)

    # Dendrogram top
    ax_dend = fig.add_subplot(gs[0, 0])
    dend = dendrogram(linkage_mat, ax=ax_dend, color_threshold=0,
                      above_threshold_color="gray", no_labels=True,
                      link_color_func=lambda k: "gray")
    ax_dend.set_axis_off()

    # Heatmap
    ax_heat = fig.add_subplot(gs[1, 0])
    order   = dend["leaves"]
    ordered_mat  = cos_mat[np.ix_(order, order)]
    ordered_names = [curated[i] for i in order]

    im = ax_heat.imshow(ordered_mat, cmap="RdYlGn", vmin=0.5, vmax=1.0,
                        aspect="auto")
    ax_heat.set_xticks(range(len(ordered_names)))
    ax_heat.set_xticklabels(ordered_names, rotation=90, fontsize=5.5)
    ax_heat.set_yticks(range(len(ordered_names)))
    ax_heat.set_yticklabels(ordered_names, fontsize=5.5)
    ax_heat.tick_params(axis="both", direction="in", length=2)

    # Colour bar
    ax_cbar = fig.add_subplot(gs[1, 1])
    plt.colorbar(im, cax=ax_cbar, label="Cosine similarity")

    # Group colour strip
    ax_strip = fig.add_subplot(gs[0, 1])
    strip = np.array([row_colors[i] for i in order])
    for j, c in enumerate(strip):
        ax_strip.add_patch(plt.Rectangle((j, 0), 1, 1, color=c))
    ax_strip.set_xlim(0, len(strip))
    ax_strip.set_ylim(0, 1)
    ax_strip.set_axis_off()

    fig.suptitle("(4) Champion Similarity Heatmap — Hierarchical Clustering\n"
                 "(colour strip = mechanical group)", fontsize=9, fontweight="bold")

    fig.savefig(out_dir / "task4_heatmap.pdf")
    fig.savefig(out_dir / "task4_heatmap.png", dpi=300)
    plt.close(fig)

    # Compute intra vs. inter group cosine similarity
    group_memberships = {c: get_group_for_champion(c) for c in curated}
    intra, inter = [], []
    for i, ci in enumerate(curated):
        for j, cj in enumerate(curated):
            if i >= j:
                continue
            sim = cos_mat[i, j]
            if (group_memberships[ci] is not None
                    and group_memberships[ci] == group_memberships[cj]):
                intra.append(sim)
            else:
                inter.append(sim)

    return {
        "mean_intra_group_cos":  float(np.mean(intra)) if intra else float("nan"),
        "mean_inter_group_cos":  float(np.mean(inter)) if inter else float("nan"),
        "intra_minus_inter_gap": float(np.mean(intra) - np.mean(inter))
                                 if intra and inter else float("nan"),
        "n_curated_champions":   len(curated),
    }


def task_role_vs_mechanic_separation(
    embeddings: dict[str, np.ndarray],
    out_dir:    Path,
) -> dict:
    """
    Task 5: Role clustering vs. mechanical clustering comparison.
    Tests whether the embedding clusters more by mechanical group
    than by traditional role label — the core geometric target.

    Computes silhouette-like score for both taxonomies and compares.
    """
    # Role assignment (approximate — extend as needed)
    ROLE_GROUPS = {
        "top":     ["Malphite", "Darius", "Nasus", "Trundle", "Jayce",
                    "Sion", "Anivia", "Galio"],
        "jungle":  ["Amumu", "Zac", "Rek'Sai", "Rengar", "Kha'Zix",
                    "Nidalee", "Elise", "Ekko"],
        "mid":     ["Veigar", "Lissandra", "Annie", "Zed", "Akali",
                    "Talon", "LeBlanc", "Orianna"],
        "adc":     ["Jinx", "Kog'Maw", "Twitch"],
        "support": ["Thresh", "Blitzcrank", "Nautilus", "Seraphine",
                    "Taric", "Lulu"],
    }

    def silhouette_score_simple(
        emb_dict: dict[str, np.ndarray],
        groups:   dict[str, list],
    ) -> float:
        """Simplified silhouette: mean(intra) - mean(inter) per champion, averaged."""
        scores = []
        all_names = [n for members in groups.values() for n in members
                     if n in emb_dict]
        name_to_group = {}
        for g, members in groups.items():
            for m in members:
                if m in emb_dict:
                    name_to_group[m] = g

        for name in all_names:
            group   = name_to_group[name]
            ingroup = [m for m in groups[group] if m in emb_dict and m != name]
            outgroup = [m for m in all_names
                        if name_to_group.get(m) != group and m != name]
            if not ingroup or not outgroup:
                continue
            a = np.mean([cosine_similarity(emb_dict[name], emb_dict[m])
                         for m in ingroup])
            b = np.mean([cosine_similarity(emb_dict[name], emb_dict[m])
                         for m in outgroup])
            scores.append(a - b)

        return float(np.mean(scores)) if scores else float("nan")

    sil_role    = silhouette_score_simple(embeddings, ROLE_GROUPS)
    sil_mechanic = silhouette_score_simple(embeddings, MECHANICAL_GROUPS)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 3))
    fig.suptitle("(5) Role vs. Mechanical Clustering Silhouette",
                 fontsize=9, fontweight="bold")

    bars = ax.bar(
        ["Role-based\nclustering", "Mechanical\nclustering"],
        [sil_role, sil_mechanic],
        color=["#d62728", "#2ca02c"],
        alpha=0.85, width=0.4,
    )
    ax.axhline(0, color="black", lw=0.6)
    for bar, val in zip(bars, [sil_role, sil_mechanic]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_ylabel("Silhouette score\n(intra − inter cosine sim)")
    ax.set_ylim(min(0, sil_role - 0.05), max(sil_mechanic + 0.05, 0.2))
    ax.grid(True, axis="y", ls=":", lw=0.4)
    ax.tick_params(direction="in")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "task5_role_vs_mechanic.pdf")
    fig.savefig(out_dir / "task5_role_vs_mechanic.png", dpi=300)
    plt.close(fig)

    return {
        "silhouette_role_clustering":      sil_role,
        "silhouette_mechanic_clustering":  sil_mechanic,
        "mechanic_advantage":              sil_mechanic - sil_role,
        "interpretation": (
            "mechanic > role → model captures cross-role similarity ✓"
            if sil_mechanic > sil_role
            else "role > mechanic → model still clusters by role ✗"
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark C2CP champion embedding model"
    )
    parser.add_argument("--embeddings",  type=str, required=True,
                        help="Path to champions_embeddings.json")
    parser.add_argument("--rationales",  type=str, default=None,
                        help="Path to champion_rationales.jsonl (optional, for reference)")
    parser.add_argument("--out",         type=str, default="./output/benchmark_c2cp/",
                        help="Output directory for figures and report")
    parser.add_argument("--top_k",       type=int, default=5,
                        help="k for nearest-neighbour precision@k (default 5)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embeddings from {args.embeddings} ...")
    embeddings = load_embeddings(args.embeddings)
    print(f"  Loaded {len(embeddings)} champion embeddings "
          f"(dim={next(iter(embeddings.values())).shape[0]})")

    results = {}

    print("\n── Task 1: Global geometry ──────────────────────────────────────")
    results["global_geometry"] = task_global_geometry(embeddings, out_dir)
    r = results["global_geometry"]
    print(f"  Effective rank:       {r['effective_rank']:.2f}")
    print(f"  Mean pairwise cos:    {r['mean_pairwise_cos']:.4f}")
    print(f"  Per-dim std:          {r['per_dim_std']:.5f}")
    print(f"  Dims for 50/80/95%:   {r['dim_50pct_variance']} / "
          f"{r['dim_80pct_variance']} / {r['dim_95pct_variance']}")
    print(f"  Dead dims (<0.01 std):{r['n_dead_dims']}")

    print("\n── Task 2: Expected close/far pairs ────────────────────────────")
    results["expected_pairs"] = task_expected_pairs(embeddings, out_dir)
    r = results["expected_pairs"]
    print(f"  Mean close sim:       {r['mean_close_sim']:.4f}")
    print(f"  Mean far sim:         {r['mean_far_sim']:.4f}")
    print(f"  Gap (close − far):    {r['gap_close_minus_far']:.4f}")
    print(f"  Close precision:      {r['close_precision']:.2%}  "
          f"({r['n_correct_close_pairs']}/{r['n_total_close_pairs']})")
    if r["missing_pairs"]:
        print(f"  Missing pairs:        {r['missing_pairs']}")

    print(f"\n── Task 3: NN precision@{args.top_k} ──────────────────────────")
    results["nn_recall"] = task_nearest_neighbours(embeddings, out_dir, args.top_k)
    r = results["nn_recall"]
    print(f"  Overall precision@{args.top_k}: {r['overall_precision_at_k']:.4f}  "
          f"(random baseline: {r['random_baseline']:.3f})")
    for group, gr in r["per_group"].items():
        print(f"    {group:<35} {gr['mean_precision_at_k']:.3f}")

    print("\n── Task 4: Similarity heatmap ───────────────────────────────────")
    results["heatmap"] = task_similarity_heatmap(embeddings, out_dir)
    r = results["heatmap"]
    print(f"  Mean intra-group cos: {r.get('mean_intra_group_cos', 'n/a'):.4f}")
    print(f"  Mean inter-group cos: {r.get('mean_inter_group_cos', 'n/a'):.4f}")
    print(f"  Intra − inter gap:    {r.get('intra_minus_inter_gap', 'n/a'):.4f}")

    print("\n── Task 5: Role vs. mechanic separation ─────────────────────────")
    results["role_vs_mechanic"] = task_role_vs_mechanic_separation(embeddings, out_dir)
    r = results["role_vs_mechanic"]
    print(f"  Silhouette (role):    {r['silhouette_role_clustering']:.4f}")
    print(f"  Silhouette (mechanic):{r['silhouette_mechanic_clustering']:.4f}")
    print(f"  Mechanic advantage:   {r['mechanic_advantage']:.4f}")
    print(f"  → {r['interpretation']}")

    # Save full results as JSON
    report_path = out_dir / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull report saved to {report_path}")
    print(f"Figures saved to     {out_dir}/")


if __name__ == "__main__":
    main()
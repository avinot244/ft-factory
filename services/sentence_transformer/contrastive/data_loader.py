"""
data_loader.py
--------------
Dataset and DataLoader utilities for contrastive champion embedding training.
Supports both dataset variants transparently:

  Dataset (A) — avinot/Champion-Similarity-v5
      Columns: anchor (str), positive (str), negatives (list[str])
      All values are raw rationale strings.

  Dataset (B) — avinot/Champion-Similarity-v6
      Columns: anchor_rationale (str), positive_rationale (str),
               negative_rationales (list[str]),
               anchor_champion (str), positive_champion (str),
               negative_champions (list[str])

The variant is detected automatically from the column names of the loaded
dataset.  No user-facing flag is needed.

Two operating modes
-------------------
TEXT mode  (cache_path=None)
    __getitem__ returns raw rationale strings.

CACHED mode  (cache_path="backbone_cache.pt")
    All unique rationale strings are encoded once with BAAI/bge-m3 and
    stored in a {rationale: Tensor(D,)} dict on disk.  __getitem__ returns
    pre-computed float32 tensors.  Training only runs the tiny MLP head.

    Build the cache with:
        ChampionSimilarityDataset.build_cache(
            dataset_id = "avinot/Champion-Similarity-v5",  # or v6
            save_path  = "cache/backbone_cache.pt",
            device     = "cuda",
        )
    Or pass auto_cache=True to the constructor.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------

def _detect_schema(dataset) -> str:
    """
    Return "v5" or "v6" based on the column names of a HuggingFace dataset split.

    v5 columns : anchor, positive, negatives
    v6 columns : anchor_rationale, positive_rationale, negative_rationales, ...
    """
    cols = set(dataset.column_names)
    if "anchor_rationale" in cols:
        return "v6"
    if "anchor" in cols:
        return "v5"
    raise ValueError(
        f"Cannot detect dataset schema from columns: {cols}.\n"
        f"Expected either v5 (anchor/positive/negatives) "
        f"or v6 (anchor_rationale/positive_rationale/negative_rationales)."
    )


def _extract_row(row: dict, schema: str) -> tuple[str, str, list[str]]:
    """
    Extract (anchor_rationale, positive_rationale, negative_rationales)
    from a raw dataset row regardless of schema version.
    """
    if schema == "v5":
        # v5: negatives is a single rationale string, not a list
        return row["anchor"], row["positive"], [row["negative"]]
    else:  # v6
        return (
            row["anchor_rationale"],
            row["positive_rationale"],
            list(row["negative_rationales"]),
        )


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_cache(
    save_path:    str,
    dataset_id:   str           = "avinot/Champion-Similarity-v6",
    device:       str           = "cuda",
    batch_size:   int           = 64,
    hf_cache_dir: Optional[str] = None,
) -> None:
    """
    Encode every unique rationale string in all splits with BAAI/bge-m3
    and save the result as a { rationale: Tensor(D,) } dict to *save_path*.

    Works with both v5 and v6 dataset schemas — the schema is detected
    automatically from column names.

    Parameters
    ----------
    save_path    : destination .pt file path
    dataset_id   : HuggingFace dataset repo ID (v5 or v6)
    device       : "cuda" or "cpu"
    batch_size   : texts encoded per forward pass (lower if OOM)
    hf_cache_dir : HuggingFace datasets cache directory
    """
    from sentence_transformers import SentenceTransformer

    print(f"[cache] Loading BAAI/bge-m3 for one-time encoding (device={device})...")
    backbone = SentenceTransformer("BAAI/bge-m3", device=device)
    backbone.eval()

    print(f"[cache] Loading dataset '{dataset_id}'...")
    raw = load_dataset(dataset_id, cache_dir=hf_cache_dir)

    # Detect schema from the first available split
    first_split = next(iter(raw))
    schema = _detect_schema(raw[first_split])
    print(f"[cache] Detected schema: {schema}")

    # Collect every unique rationale string across all splits
    unique_texts: set[str] = set()
    for split in raw:
        for row in raw[split]:
            anc, pos, negs = _extract_row(row, schema)
            unique_texts.add(anc)
            unique_texts.add(pos)
            unique_texts.update(negs)

    unique_texts_list = list(unique_texts)
    N = len(unique_texts_list)
    print(f"[cache] {N} unique rationales to encode (batch_size={batch_size})...")

    parent = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(parent, exist_ok=True)

    # Encode in batches, flushing each shard to disk immediately so RAM
    # never holds more than one batch worth of tensors at a time.
    n_batches = (N + batch_size - 1) // batch_size
    shard_dir = save_path + ".shards"
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths: list[str] = []

    for batch_idx in range(n_batches):
        start       = batch_idx * batch_size
        end         = min(start + batch_size, N)
        batch_texts = unique_texts_list[start:end]

        with torch.no_grad():
            batch_embs = backbone.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )                                       # (end-start, D) on device

        shard: dict[str, torch.Tensor] = {
            text: emb.cpu().float()
            for text, emb in zip(batch_texts, batch_embs)
        }
        shard_path = os.path.join(shard_dir, f"shard_{batch_idx:05d}.pt")
        torch.save(shard, shard_path)
        shard_paths.append(shard_path)

        del batch_embs, shard
        if device == "cuda":
            torch.cuda.empty_cache()

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"[cache]   {end}/{N} rationales encoded and flushed...")

    # Merge shards one at a time — never holds more than one shard in RAM
    print("[cache] Merging shards...")
    final_cache: dict[str, torch.Tensor] = {}
    for shard_path in shard_paths:
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        final_cache.update(shard)
        del shard
        os.remove(shard_path)

    torch.save(final_cache, save_path)
    os.rmdir(shard_dir)
    print(f"[cache] Saved {len(final_cache)} embeddings → {save_path}")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChampionSimilarityDataset(Dataset):
    """
    Wraps one split of a champion similarity dataset (v5 or v6).
    The schema is detected automatically from column names.

    Parameters
    ----------
    dataset_id   : HuggingFace repo ID — v5 or v6
    split        : "train" | "validation" | "test"
    cache_dir    : HuggingFace dataset cache directory
    cache_path   : path to a pre-built backbone embedding cache (.pt).
                   When provided and the file exists → CACHED mode
                   (__getitem__ returns tensors).
    auto_cache   : if True and cache_path is missing, build it automatically.
    cache_device : device used for encoding when auto_cache=True
    cache_batch  : encoding batch size used when auto_cache=True
    """

    @staticmethod
    def build_cache(
        save_path:    str,
        dataset_id:   str           = "avinot/Champion-Similarity-v6",
        device:       str           = "cuda",
        batch_size:   int           = 64,
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        """Convenience wrapper around the module-level build_cache()."""
        build_cache(
            save_path    = save_path,
            dataset_id   = dataset_id,
            device       = device,
            batch_size   = batch_size,
            hf_cache_dir = hf_cache_dir,
        )

    def __init__(
        self,
        dataset_id:   str           = "avinot/Champion-Similarity-v6",
        split:        str           = "train",
        cache_dir:    Optional[str] = None,
        cache_path:   Optional[str] = None,
        auto_cache:   bool          = False,
        cache_device: str           = "cuda",
        cache_batch:  int           = 64,
    ) -> None:
        super().__init__()
        self.split      = split
        self.dataset_id = dataset_id

        raw: DatasetDict = load_dataset(dataset_id, cache_dir=cache_dir)
        if split not in raw:
            raise ValueError(
                f"Split '{split}' not found in '{dataset_id}'. "
                f"Available: {list(raw.keys())}"
            )
        self._data   = raw[split]
        self._schema = _detect_schema(self._data)
        print(f"[ChampionSimilarityDataset] '{dataset_id}' / '{split}' "
              f"— schema={self._schema}, {len(self._data)} samples.")

        # -- Embedding cache ------------------------------------------------
        self._emb_cache: Optional[dict[str, torch.Tensor]] = None

        if cache_path is not None:
            if not os.path.exists(cache_path):
                if auto_cache:
                    print(f"[ChampionSimilarityDataset] Cache not found at "
                          f"'{cache_path}' — building automatically...")
                    build_cache(
                        save_path    = cache_path,
                        dataset_id   = dataset_id,
                        device       = cache_device,
                        batch_size   = cache_batch,
                        hf_cache_dir = cache_dir,
                    )
                else:
                    raise FileNotFoundError(
                        f"Backbone cache not found: '{cache_path}'.\n"
                        f"Call ChampionSimilarityDataset.build_cache("
                        f"dataset_id='{dataset_id}', save_path='{cache_path}') "
                        f"first, or pass auto_cache=True."
                    )

            print(f"[ChampionSimilarityDataset] Loading cache '{cache_path}'...")
            self._emb_cache = torch.load(
                cache_path, map_location="cpu", weights_only=True
            )
            print(f"[ChampionSimilarityDataset] Cache loaded — "
                  f"{len(self._emb_cache)} entries.")

    @property
    def cached(self) -> bool:
        """True when operating in CACHED (tensor) mode."""
        return self._emb_cache is not None

    @property
    def schema(self) -> str:
        """'v5' or 'v6'."""
        return self._schema

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(
        self,
        idx: int,
    ) -> Union[
        tuple[str,          str,          list[str]],           # TEXT mode
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],  # CACHED mode
    ]:
        row = self._data[idx]
        anc_text, pos_text, neg_texts = _extract_row(row, self._schema)

        if self._emb_cache is None:
            return anc_text, pos_text, neg_texts

        # CACHED mode — look up pre-computed backbone tensors
        anc_emb  = self._emb_cache[anc_text]
        pos_emb  = self._emb_cache[pos_text]
        neg_embs = [self._emb_cache[t] for t in neg_texts]
        return anc_emb, pos_emb, neg_embs


# ---------------------------------------------------------------------------
# Collator  (mode-aware, schema-agnostic)
# ---------------------------------------------------------------------------

class ContrastiveCollator:
    """
    Pads the variable-length negative list to K_max within each batch.
    Detects TEXT vs CACHED mode from the type of the first sample.

    TEXT mode output
    ----------------
    anchors   : list[str]           (B,)
    positives : list[str]           (B,)
    negatives : list[list[str]]     (B, K_max)  — padded with ""
    neg_mask  : BoolTensor          (B, K_max)

    CACHED mode output
    ------------------
    anc_emb  : FloatTensor   (B, D)
    pos_emb  : FloatTensor   (B, D)
    neg_emb  : FloatTensor   (B, K_max, D)  — padding positions are zeros
    neg_mask : BoolTensor    (B, K_max)
    """

    def __call__(self, batch):
        anchors, positives, negatives_list = zip(*batch)

        is_cached = isinstance(anchors[0], torch.Tensor)
        k_max     = max(len(negs) for negs in negatives_list)

        mask_rows = [
            [True] * len(negs) + [False] * (k_max - len(negs))
            for negs in negatives_list
        ]
        neg_mask = torch.tensor(mask_rows, dtype=torch.bool)

        if not is_cached:
            padded_negatives = [
                list(negs) + [""] * (k_max - len(negs))
                for negs in negatives_list
            ]
            return list(anchors), list(positives), padded_negatives, neg_mask

        D = anchors[0].shape[0]
        anc_emb = torch.stack(list(anchors))
        pos_emb = torch.stack(list(positives))

        neg_rows = []
        for negs in negatives_list:
            pad = [torch.zeros(D)] * (k_max - len(negs))
            neg_rows.append(torch.stack(list(negs) + pad))
        neg_emb = torch.stack(neg_rows)

        return anc_emb, pos_emb, neg_emb, neg_mask


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset:     ChampionSimilarityDataset,
    batch_size:  int  = 32,
    shuffle:     bool = True,
    num_workers: int  = 0,
    pin_memory:  bool = False,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = ContrastiveCollator(),
        num_workers = num_workers,
        pin_memory  = pin_memory and dataset.cached,
    )
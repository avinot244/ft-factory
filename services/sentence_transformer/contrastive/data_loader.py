from __future__ import annotations

import os
from typing import Optional, Union

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict


# ---------------------------------------------------------------------------
# Cache builder
# ---------------------------------------------------------------------------

def build_cache(
    save_path:    str,
    device:       str           = "cuda",
    batch_size:   int           = 64,
    hf_cache_dir: Optional[str] = None,
) -> None:
    """
    Encode every unique text in all three splits with BAAI/bge-m3 and save
    the result as a { text: Tensor(D,) } dict to *save_path*.
 
    This is a one-time operation.  Texts are encoded one batch at a time and
    accumulated directly into the output dict so that neither the full
    embedding matrix nor the raw text list need to live in RAM simultaneously.
 
    Parameters
    ----------
    save_path    : destination .pt file path
    device       : "cuda" or "cpu"
    batch_size   : number of texts encoded per forward pass (lower if OOM)
    hf_cache_dir : HuggingFace datasets cache directory
    """
    from sentence_transformers import SentenceTransformer
 
    print("[cache] Loading BAAI/bge-m3 for one-time encoding...")
    backbone = SentenceTransformer("BAAI/bge-m3", device=device)
    backbone.eval()
 
    print("[cache] Loading dataset from HuggingFace...")
    raw = load_dataset("avinot/Champion-Similarity-v6", cache_dir=hf_cache_dir)
 
    # Collect every unique rationale string across all splits
    unique_texts: set[str] = set()
    for split in raw:
        for row in raw[split]:
            unique_texts.add(row["anchor_rationale"])
            unique_texts.add(row["positive_rationale"])
            unique_texts.update(row["negative_rationales"])
 
    unique_texts_list = list(unique_texts)
    N = len(unique_texts_list)
    print(f"[cache] {N} unique texts to encode "
          f"(batch_size={batch_size}, device={device})...")
 
    parent = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(parent, exist_ok=True)
 
    # Encode one batch at a time and flush each shard to disk immediately.
    # The in-memory dict never holds more than one batch worth of tensors.
    n_batches = (N + batch_size - 1) // batch_size
    shard_dir = save_path + ".shards"
    os.makedirs(shard_dir, exist_ok=True)
    shard_paths: list[str] = []
 
    for batch_idx in range(n_batches):
        start       = batch_idx * batch_size
        end         = min(start + batch_size, N)
        batch_texts = unique_texts_list[start:end]  # exactly (end-start) strings
 
        with torch.no_grad():
            batch_embs = backbone.encode(
                batch_texts,
                batch_size=len(batch_texts),
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )                                       # (end-start, D) on device
 
        # Build a small shard dict and immediately flush to disk
        shard: dict[str, torch.Tensor] = {
            text: emb.cpu().float()
            for text, emb in zip(batch_texts, batch_embs)
        }
        shard_path = os.path.join(shard_dir, f"shard_{batch_idx:05d}.pt")
        torch.save(shard, shard_path)
        shard_paths.append(shard_path)
 
        # Free GPU and CPU memory before the next batch
        del batch_embs, shard
        if device == "cuda":
            torch.cuda.empty_cache()
 
        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            print(f"[cache]   {end}/{N} texts encoded and flushed to disk...")
 
    # Merge shards one at a time into the final file, deleting each shard
    # after it is consumed so RAM never holds more than one shard at once.
    print("[cache] Merging shards into final cache file...")
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
    Wraps one split of avinot/Champion-Similarity-v6.

    Parameters
    ----------
    split        : "train" | "validation" | "test"
    cache_dir    : HuggingFace dataset cache directory
    cache_path   : path to a pre-built backbone embedding cache (.pt).
                   When provided and the file exists, the dataset operates
                   in CACHED mode and __getitem__ returns tensors instead
                   of strings.
    auto_cache   : if True and cache_path points to a non-existent file,
                   build the cache automatically before loading.
    cache_device : device used for encoding when auto_cache=True
    cache_batch  : encoding batch size used when auto_cache=True
    """

    @staticmethod
    def build_cache(
        save_path:    str,
        device:       str           = "cuda",
        batch_size:   int           = 64,
        hf_cache_dir: Optional[str] = None,
    ) -> None:
        """Convenience wrapper around the module-level build_cache()."""
        build_cache(
            save_path    = save_path,
            device       = device,
            batch_size   = batch_size,
            hf_cache_dir = hf_cache_dir,
        )

    # ------------------------------------------------------------------

    def __init__(
        self,
        split:        str           = "train",
        cache_dir:    Optional[str] = None,
        cache_path:   Optional[str] = None,
        auto_cache:   bool          = False,
        cache_device: str           = "cuda",
        cache_batch:  int           = 64,
    ) -> None:
        super().__init__()
        self.split = split

        raw: DatasetDict = load_dataset(
            "avinot/Champion-Similarity-v6",
            cache_dir=cache_dir,
        )
        if split not in raw:
            raise ValueError(
                f"Split '{split}' not found. Available: {list(raw.keys())}"
            )
        self._data = raw[split]

        # -- Embedding cache -----------------------------------------------
        self._emb_cache: Optional[dict[str, torch.Tensor]] = None

        if cache_path is not None:
            if not os.path.exists(cache_path):
                if auto_cache:
                    print(f"[ChampionSimilarityDataset] Cache not found at "
                          f"'{cache_path}' — building automatically...")
                    build_cache(
                        save_path    = cache_path,
                        device       = cache_device,
                        batch_size   = cache_batch,
                        hf_cache_dir = cache_dir,
                    )
                else:
                    raise FileNotFoundError(
                        f"Backbone cache not found: '{cache_path}'.\n"
                        f"Run ChampionSimilarityDataset.build_cache('{cache_path}') "
                        f"first, or pass auto_cache=True."
                    )

            print(f"[ChampionSimilarityDataset] Loading embedding cache "
                  f"from '{cache_path}'...")
            self._emb_cache = torch.load(cache_path, map_location="cpu",
                                         weights_only=True)
            print(f"[ChampionSimilarityDataset] Cache loaded — "
                  f"{len(self._emb_cache)} entries.")

    # ------------------------------------------------------------------

    @property
    def cached(self) -> bool:
        """True when operating in CACHED (tensor) mode."""
        return self._emb_cache is not None

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(
        self,
        idx: int,
    ) -> Union[
        tuple[str,          str,          list[str]],            # TEXT mode
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],  # CACHED mode
    ]:
        row = self._data[idx]
        anchor_text    = row["anchor_rationale"]
        positive_text  = row["positive_rationale"]
        negative_texts = row["negative_rationales"]

        if self._emb_cache is None:
            # TEXT mode
            return anchor_text, positive_text, negative_texts

        # CACHED mode — look up pre-computed backbone tensors
        anc_emb  = self._emb_cache[anchor_text]                     # (D,)
        pos_emb  = self._emb_cache[positive_text]                   # (D,)
        neg_embs = [self._emb_cache[t] for t in negative_texts]    # list[(D,)]
        return anc_emb, pos_emb, neg_embs


# ---------------------------------------------------------------------------
# Collator  (mode-aware)
# ---------------------------------------------------------------------------

class ContrastiveCollator:
    """
    Pads the variable-length negative list to K_max within each batch.
    Automatically detects TEXT vs CACHED mode from the type of the first sample.

    TEXT mode output
    ----------------
    anchors   : list[str]           (B,)
    positives : list[str]           (B,)
    negatives : list[list[str]]     (B, K_max)  — padded with ""
    neg_mask  : BoolTensor          (B, K_max)  True = real negative

    CACHED mode output
    ------------------
    anc_emb  : FloatTensor   (B, D)
    pos_emb  : FloatTensor   (B, D)
    neg_emb  : FloatTensor   (B, K_max, D)  — padding positions are zeros
    neg_mask : BoolTensor    (B, K_max)     True = real negative
    """

    def __call__(self, batch):
        anchors, positives, negatives_list = zip(*batch)

        is_cached = isinstance(anchors[0], torch.Tensor)
        k_max     = max(len(negs) for negs in negatives_list)

        # Mask is the same for both modes
        mask_rows = [
            [True]  * len(negs) + [False] * (k_max - len(negs))
            for negs in negatives_list
        ]
        neg_mask = torch.tensor(mask_rows, dtype=torch.bool)   # (B, K_max)

        if not is_cached:
            # TEXT mode — pad with empty strings
            padded_negatives = [
                list(negs) + [""] * (k_max - len(negs))
                for negs in negatives_list
            ]
            return list(anchors), list(positives), padded_negatives, neg_mask

        # CACHED mode — stack into dense tensors; pad with zero vectors
        D = anchors[0].shape[0]

        anc_emb = torch.stack(list(anchors))    # (B, D)
        pos_emb = torch.stack(list(positives))  # (B, D)

        neg_rows = []
        for negs in negatives_list:
            pad   = [torch.zeros(D)] * (k_max - len(negs))
            neg_rows.append(torch.stack(list(negs) + pad))  # (K_max, D)
        neg_emb = torch.stack(neg_rows)          # (B, K_max, D)

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
    """
    Build a DataLoader with the mode-aware ContrastiveCollator attached.

    In CACHED mode, num_workers >= 2 and pin_memory=True are safe and
    recommended — the heavy backbone encoding is fully offline so there
    are no CUDA-in-worker issues.
    """
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = shuffle,
        collate_fn  = ContrastiveCollator(),
        num_workers = num_workers,
        pin_memory  = pin_memory and dataset.cached,
    )
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset, DatasetDict

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChampionSimilarityDataset(Dataset):
    """
    Wraps one split of avinot/Champion-Similarity-v6.

    Parameters
    ----------
    split : "train" | "validation" | "test"
    cache_dir : optional path for HuggingFace dataset cache
    """

    def __init__(
        self,
        split: str = "train",
        cache_dir: Optional[str] = None,
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

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[str, str, list[str]]:
        """
        Returns
        -------
        anchor_text   : str
        positive_text : str
        negative_texts: list[str]   variable length 1..9
        """
        row = self._data[idx]
        anchor_text   = row["anchor_rationale"]
        positive_text = row["positive_rationale"]
        negative_texts = row["negative_rationales"]
        return anchor_text, positive_text, negative_texts


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

class ContrastiveCollator:
    """
    Pads the variable-length negative list to K_max within each batch
    and builds a boolean mask of valid (non-padding) positions.

    Output
    ------
    anchor_texts   : list[str]         (B,)
    positive_texts : list[str]         (B,)
    negative_texts : list[list[str]]   (B, K_max)  — padded rows contain ""
    neg_mask       : BoolTensor        (B, K_max)  True = real negative
    """

    _PAD = ""

    def __call__(
        self,
        batch: list[tuple[str, str, list[str]]],
    ) -> tuple[list[str], list[str], list[list[str]], torch.BoolTensor]:
        anchors, positives, negatives_list = zip(*batch)

        k_max = max(len(negs) for negs in negatives_list)

        padded_negatives: list[list[str]] = []
        mask_rows: list[list[bool]] = []

        for negs in negatives_list:
            k = len(negs)
            pad = k_max - k
            padded_negatives.append(negs + [self._PAD] * pad)
            mask_rows.append([True] * k + [False] * pad)

        neg_mask = torch.tensor(mask_rows, dtype=torch.bool)  # (B, K_max)

        return list(anchors), list(positives), padded_negatives, neg_mask


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: ChampionSimilarityDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """
    Build a DataLoader with the contrastive collator already attached.

    Parameters
    ----------
    dataset     : ChampionSimilarityDataset instance
    batch_size  : mini-batch size
    shuffle     : True for training, False for eval
    num_workers : passed to DataLoader
    pin_memory  : passed to DataLoader (useful when GPU is available)

    Returns
    -------
    DataLoader that yields (anchor_texts, positive_texts, negative_texts, neg_mask)
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=ContrastiveCollator(),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
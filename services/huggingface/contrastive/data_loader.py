from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from typing import Literal

from utils.token_manager import get_hf_token


class TripletDataset(Dataset):
    def __init__(self, split: Literal["train", "validation", "test"]):
        raw = load_dataset(
            "avinot/Champion-Similarity-v6",
            token=get_hf_token("read"),
            split=split,
        )
        # Schema (v6 updated): flat top-level columns — champion names and
        # rationales are stored in separate fields rather than nested dicts:
        #   anchor_champion      str
        #   anchor_rationale     str
        #   positive_champion    str
        #   positive_rationale   str
        #   negative_champions   list[str]
        #   negative_rationales  list[str]
        #
        # Only rationales are passed to the model so the encoder learns
        # purely from behavioural text and generalises to arbitrary plain-text
        # inputs at inference time without any structural format dependency.
        self.anchors   = raw["anchor_rationale"]    # list[str]
        self.positives = raw["positive_rationale"]  # list[str]
        self.negatives = raw["negative_rationales"] # list[list[str]]

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx):
        return (
            self.anchors[idx],
            self.positives[idx],
            self.negatives[idx],  # list[str], variable length 1-9
        )


def collate_fn(batch):
    """
    Custom collate that keeps per-sample negative lists intact.
    Returns:
        anchors   : list[str]        length B
        positives : list[str]        length B
        negatives : list[list[str]]  shape (B, K_i) — K_i may differ across rows
    """
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )
from datasets import load_dataset
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from typing import Literal
import torch

from utils.token_manager import get_hf_token

class TripletDataset(Dataset):
    def __init__(self, split : Literal["train", "validation"]):
        dataframe = pd.DataFrame(load_dataset("avinot/Champion-Similarity-v2", token=get_hf_token("read"), split=split))
        self.anchor = dataframe["anchor"].to_numpy()
        self.positive = dataframe["positive"].to_numpy()
        self.negative = dataframe["negative"].to_numpy()

    def __len__(self):
        return len(self.anchor)
    
    def __getitem__(self, idx):
        return (
            self.anchor[idx],
            self.positive[idx],
            self.negative[idx]
        )
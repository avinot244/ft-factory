from datasets import load_dataset
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from typing import Literal
import torch

from utils.token_manager import get_hf_token

class TripletDataset(Dataset):
    def __init__(self, split : Literal["train", "validation"], tokenizer : AutoTokenizer):
        dataframe = pd.DataFrame(load_dataset("avinot/Champion-Similarity", token=get_hf_token("read"), split=split))
        self.tokenizer = tokenizer
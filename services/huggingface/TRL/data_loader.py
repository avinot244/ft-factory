from typing import Literal
from datasets import load_dataset
from datasets.arrow_dataset import Dataset

from utils.token_manager import get_hf_token

def data_loader(split : Literal["train", "validation"]) -> Dataset:
    dataset = load_dataset("avinot/LoL-instruct", split=split, token=get_hf_token("read"))
    dataset = dataset.remove_columns(["id", "label"])

    return dataset
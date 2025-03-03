from datasets import load_dataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datasets.arrow_dataset import Dataset
from typing import Literal

def data_loader(tokenizer : PreTrainedTokenizerFast, split : Literal["train", "validation"]) -> Dataset:
    dataset = load_dataset("avinot/LoL-Champion-Corpus-v2", split=split)
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples : dict):
        return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
    
    

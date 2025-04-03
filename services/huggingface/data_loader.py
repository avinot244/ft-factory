from datasets import load_dataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datasets.arrow_dataset import Dataset
from typing import Literal
from utils.token_manager import get_hf_token

from utils.globals import BLOCK_SIZE

def data_loader(tokenizer : PreTrainedTokenizerFast, split : Literal["train", "validation"]) -> Dataset:
    
    ds = load_dataset("avinot/LoL-Corpus", split=split, token=get_hf_token("read"))
    ds = ds.flatten()
    
    # First we tokenize the dataset
    def preprocess_function(examples):
        return tokenizer([" ".join(x) for x in examples["text"]])
    
    tokenized_ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=ds.column_names
    )
    
    # Then we group the texts into fixed size of BLOCK_SIZE
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    
    lm_dataset = tokenized_ds.map(group_texts, batched=True, num_proc=4)
    
    return lm_dataset

def data_loader_eli5(tokenizer : PreTrainedTokenizerFast) -> Dataset:
    eli5 = load_dataset("eli5_category", split="train[:5000]")
    eli5 = eli5.train_test_split(test_size=0.2).flatten()
    
    def preprosess_function(examples):
        return tokenizer([" ".join(x) for x in examples["answers.text"]])

    tokenized_eli5 = eli5.map(
        preprosess_function,
        batched=True,
        num_proc=4,
        remove_columns=eli5["train"].column_names
    )
    
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= BLOCK_SIZE:
            total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
    return lm_dataset
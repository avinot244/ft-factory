from datasets import load_dataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datasets.arrow_dataset import Dataset




def data_loader(tokenizer : PreTrainedTokenizerFast) -> Dataset:
    dataset = load_dataset("avinot/LoL-Champion-Corpus-v2", split="train")
    EOS_TOKEN = tokenizer.eos_token
    
    def formatting_prompts_func(examples : dict):
        return { "text" : [example + EOS_TOKEN for example in examples["text"]] }
    
    dataset = dataset.map(formatting_prompts_func, batched=True)
    return dataset
    
    

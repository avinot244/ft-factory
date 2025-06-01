from transformers import (
    PreTrainedModel, 
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch
from peft import PeftModel
from typing import Tuple

def get_model_and_tokenizer_TRL() -> Tuple[PreTrainedModel, AutoTokenizer]:
    # Load the base model with memory-efficient settings and specify dtype
    # Explicitly load in float16 or bfloat16
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, # Prefer bfloat16 if supported
        device_map="auto",       # Automatically map model to available devices
        # Add trust_remote_code=True if necessary
    )
    model : PeftModel = PeftModel.from_pretrained(base_model, "avinot/Lollama3.2-1B-lora-3ep-v3")
    model = model.merge_and_unload()

    # Set requires_grad=True for all parameters after merging
    for param in model.parameters():
        param.requires_grad = True

    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()

    # Set model to training mode
    model.train()

    for param in model.parameters():
        param.requires_grad = True

    # Load the instruct tokenizer - essential for handling the chat format
    tokenizer : AutoTokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-instruct")

    # Set pad token if not already set (common for chat models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # Also set padding side
        tokenizer.padding_side = "right" # Llama models typically prefer right padding
        
    return (model, tokenizer)
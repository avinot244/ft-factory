from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

from utils.token_manager import get_hf_token


# Model loader
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(base_model, "avinot/Lollama3.2-1B-lora-3ep-v3")
model = model.merge_and_unload()

# Set requires_grad=True for all parameters after merging
for param in model.parameters():
    param.requires_grad = True

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Set model to training mode
model.train()

model, tokenizer = setup_chat_format(model, tokenizer)


# Data loader
train_dataset = load_dataset("avinot/LoL-instruct", split="train")
train_dataset = train_dataset.remove_columns(["id", "label"])
validation_dataset = load_dataset("avinot/LoL-instruct", split="validation")
validation_dataset = validation_dataset.remove_columns(["id", "label"])

def formatting_prompts_func(example):
    return f"### Question: {example['prompt']}\n ### Answer: {example['completion']}"

training_args = SFTConfig(
    output_dir="./results/LoLlama3.2-1B-lora-3ep-v3-instruct"
)

trainer = SFTTrainer(
    model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    # formatting_func=formatting_prompts_func
)

trainer.train()
trainer.push_to_hub(token=get_hf_token("write"))
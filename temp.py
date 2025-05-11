from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, get_peft_model
import os # Import os for environment variable

from utils.token_manager import get_hf_token # Assuming this utility exists

# Set the Transformers cache directory (optional but recommended)
# os.environ['TRANSFORMERS_CACHE'] = '/path/to/your/cache'

# Data loader
# The dataset is already in the correct message format
train_dataset = load_dataset("avinot/LoL-instruct", split="train")
train_dataset = train_dataset.remove_columns(["id", "label"])

validation_dataset = load_dataset("avinot/LoL-instruct", split="validation")
validation_dataset = validation_dataset.remove_columns(["id", "label"])

# Model loader
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
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-instruct")

# Set pad token if not already set (common for chat models)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    # Also set padding side
    tokenizer.padding_side = "right" # Llama models typically prefer right padding


training_args = SFTConfig(
    output_dir="./results/LoLlama-3.2-1B-lora-3ep-v3-instruct", # Output directory name
    num_train_epochs=3,
    per_device_train_batch_size=2, # Keep small for memory
    per_device_eval_batch_size=2,   # Keep small
    gradient_accumulation_steps=8, # Use accumulation to simulate a larger batch size
    # --- Try a significantly lower learning rate to check for instability ---
    # learning_rate=5e-7, # Reduced learning rate as a test
    learning_rate=2e-5,
    # ----------------------------------------------------------------------
    logging_steps=5,
    packing=False,
    # No peft_config for full fine-tuning
    gradient_checkpointing=True, # Keep enabled for memory
    # --- Enable mixed precision for numerical stability and memory ---
    # Match the dtype you loaded the model with, prefer bf16 if supported
    bf16=torch.cuda.is_bf16_supported(), # Set to True if BF16 is supported
    fp16=not torch.cuda.is_bf16_supported(), # Otherwise set to True for FP16
    # -----------------------------------------------------------------
    # Add evaluation strategy
    evaluation_strategy="steps",
    eval_steps=50, # Evaluate every 50 steps (adjust as needed)
    save_steps=50, # Save checkpoint every 50 steps (adjust as needed)
    load_best_model_at_end=True, # Load the best model based on eval_loss at the end
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    # report_to="wandb", # Optional
    # run_name="Llama-3.2-1B-full-instruct-ft", # Optional
)

trainer = SFTTrainer(
    model=model, # Pass the base model
    tokenizer=tokenizer, # Pass the configured tokenizer
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    # Data is in 'messages' column and tokenizer has chat template, no extra formatting needed
)

print("Starting full instruction fine-tuning of base model...")
trainer.train(resume_from_checkpoint = True)
print("Training finished.")

# Save the fully fine-tuned model
print("Saving model...")
trainer.save_model()
print("Model saved.")

# Push the fully fine-tuned model to the hub
print("Pushing to hub...")
# Ensure you are pushing to a new repository or have permissions to overwrite
# trainer.push_to_hub(repo_id="your-username/your-new-full-ft-repo", token=get_hf_token("write"))
trainer.push_to_hub(token=get_hf_token("write"))
print("Pushed to hub.")
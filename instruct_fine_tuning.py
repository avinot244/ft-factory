from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, LoraConfig, get_peft_model
import os # Import os for environment variable

from utils.token_manager import get_hf_token # Assuming this utility exists

# Set the Transformers cache directory (optional but recommended)
# os.environ['TRANSFORMERS_CACHE'] = '/path/to/your/cache'

from services.huggingface.TRL.data_loader import data_loader
from services.huggingface.TRL.model_loader import get_model_and_tokenizer_TRL
from services.huggingface.TRL.trainer_loader import trainer_TRL

# Data loader
# The dataset is already in the correct message format
train_dataset = data_loader("train")
validation_dataset = data_loader("validation")


# Model loader
(model, tokenizer) = get_model_and_tokenizer_TRL()


trainer = trainer_TRL(
    model,
    tokenizer,
    train_dataset,
    validation_dataset,
    output_dir="./results/LoLlama-3.2-1B-lora-3ep-v3-instruct"
)

print("Starting full instruction fine-tuning of base model...")
trainer.train(resume_from_checkpoint = False)
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
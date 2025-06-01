from transformers import (
    PreTrainedModel, 
    AutoTokenizer,
    AutoModelForCausalLM
)
import torch
from peft import PeftModel
from typing import Tuple
from datasets.arrow_dataset import Dataset
from trl import SFTConfig, SFTTrainer

def trainer_TRL(
    model : PreTrainedModel,
    tokenizer : AutoTokenizer,
    dataset_train : Dataset,
    dataset_validation : Dataset,
    output_dir : str
) -> SFTTrainer:
    training_args = SFTConfig(
        output_dir=output_dir, # Output directory name
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
        eval_strategy="steps",
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
        processing_class=tokenizer, # Pass the configured tokenizer
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        # Data is in 'messages' column and tokenizer has chat template, no extra formatting needed
    )
    
    return trainer
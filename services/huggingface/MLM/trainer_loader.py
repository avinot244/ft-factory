from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoModelForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from datasets.arrow_dataset import Dataset
from peft.peft_model import PeftModelForCausalLM

from typing import Union

from utils.globals import *
from utils.token_manager import get_hf_token

def trainer_MLM(
    model_name : str,
    model : AutoModelForCausalLM,
    tokenizer : PreTrainedTokenizerFast,
    dataset_train : Dataset,
    dataset_validation : Dataset,
    epochs : int = EPOCHS
) -> Trainer:
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    
    
    training_args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=epochs,
        weight_decay=0.01,
        push_to_hub=True,
        hub_token=get_hf_token("write"),
        per_device_train_batch_size=2,  # Reduce batch size to save memory
        per_device_eval_batch_size=2,
        fp16=True,  # Enable mixed precision training
        gradient_accumulation_steps=4  # Accumulate gradients to simulate larger batch
        
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        data_collator=data_collator
    )
    
    return trainer
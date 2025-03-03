from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling, AutoTokenizer
from datasets.arrow_dataset import Dataset
from peft.peft_model import PeftModelForCausalLM

from utils.globals import *

def trainer_hf(model : PeftModelForCausalLM, tokenizer : AutoTokenizer, dataset_train : Dataset, dataset_validation : Dataset) -> Trainer:
    trainings_args = TrainingArguments(
        output_dir=f"./results/{MODEL_NAME_HF}",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=trainings_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_validation,
        data_collator=data_collator
    )
        
    return trainer
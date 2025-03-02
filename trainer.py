from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments

from datasets.arrow_dataset import Dataset
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from utils.globals import MAX_SEQ_LENGTH

def trainer(model : LlamaForCausalLM, tokenizer : PreTrainedTokenizerFast, dataset : Dataset):

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=8,
        args=UnslothTrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 8,

            warmup_ratio = 0.1,
            num_train_epochs = 1,

            learning_rate = 5e-5,
            embedding_learning_rate = 5e-6,

            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.00,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Use this for WandB etc
        )
        
    )
    
    trainer_stats = trainer.train()
    print(type(trainer_stats))
    return trainer_stats
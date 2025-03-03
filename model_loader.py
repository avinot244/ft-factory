from unsloth import FastLanguageModel
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft.peft_model import PeftModelForCausalLM

from utils.token_manager import get_hf_token
from utils.globals import *
import huggingface_hub


def get_model_and_tokenizer_unsloth() -> tuple[LlamaForCausalLM, PreTrainedTokenizerFast]:
    print("GET LLM MODEL")
    model : LlamaForCausalLM
    tokenizer : PreTrainedTokenizerFast
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME_US,
        max_seq_length = MAX_SEQ_LENGTH,
        dtype = DTYPE,
        load_in_4bit = LOAD_IN_4BIT,
        token = get_hf_token("read")
    )


    print("GET ASOCIATED LORA MODEL")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 128, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",
                          "embed_tokens", "lm_head",], # Add for continual pretraining
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = True,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
   
    return model, tokenizer

def get_model_and_tokenizer_hf() -> tuple[PeftModelForCausalLM, AutoTokenizer]:
    huggingface_hub.login(token=get_hf_token("read"))
    model : AutoModelForCausalLM
    tokenizer : AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME_HF)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_HF)
    lora_config = LoraConfig(
        r = 128,
        lora_alpha=32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",
                          "embed_tokens", "lm_head",],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    return model, tokenizer
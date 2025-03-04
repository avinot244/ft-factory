from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from peft import get_peft_model, LoraConfig, TaskType
from peft.peft_model import PeftModelForCausalLM

from utils.token_manager import get_hf_token
from utils.globals import *
import huggingface_hub
from typing import Literal


def get_model_and_tokenizer_hf(ft_type : Literal["classic", "lora"]) -> tuple[PeftModelForCausalLM, PreTrainedTokenizerFast]:
    huggingface_hub.login(token=get_hf_token("read"))
    model : AutoModelForCausalLM
    tokenizer : PreTrainedTokenizerFast
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_HF,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_HF)
    
    if ft_type == "lora":
        lora_config = LoraConfig(
            r = 128,
            lora_alpha=32,
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                            "embed_tokens", "lm_head",],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, lora_config)
    return model, tokenizer
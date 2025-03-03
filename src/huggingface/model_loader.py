from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from peft.peft_model import PeftModelForCausalLM

from utils.token_manager import get_hf_token
from utils.globals import *
import huggingface_hub


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
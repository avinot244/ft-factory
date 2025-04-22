from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from peft import get_peft_model, LoraConfig, TaskType
from peft.peft_model import PeftModelForCausalLM

from utils.token_manager import get_hf_token
from utils.globals import *
import huggingface_hub
from typing import Literal


def get_model_and_tokenizer_hf(model_name : str) -> tuple[AutoModelForCausalLM, PreTrainedTokenizerFast]:
    huggingface_hub.login(token=get_hf_token("read"))
    model : AutoModelForCausalLM
    tokenizer : PreTrainedTokenizerFast
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        use_cache=False
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer
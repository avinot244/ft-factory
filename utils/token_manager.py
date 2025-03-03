import json
from typing import Literal

from utils.globals import HF_TOKEN_PATH

def get_hf_token(mode : Literal["read", "write"]) -> str:
    with open(HF_TOKEN_PATH, "r") as f:
        token = json.load(f)
        return token[mode]
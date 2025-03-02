import json

from utils.globals import HF_TOKEN_PATH

def get_hf_token(mode : str) -> str:
    assert mode in ["read", "write"]
    with open(HF_TOKEN_PATH, "r") as f:
        token = json.load(f)
        return token[mode]
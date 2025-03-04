HF_TOKEN_PATH = "./tokens/hf_tokens.json"

# MODEL_NAME_HF = "meta-llama/Llama-3.2-3B" : 258h
MODEL_NAME_HF = "meta-llama/Llama-3.2-1B"
MODEL_NAME_US = "unsloth/llama-3-8b-bnb-4bit"

MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
BLOCK_SIZE = 512
EPOCHS = 10

FOURBIT_MODELS = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      # New Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           # Llama-3 15 trillion tokens model 2x faster!
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        # Phi-3 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             # Gemma 2.2x faster!
]
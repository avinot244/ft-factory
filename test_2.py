from transformers import pipeline
import torch

from utils.token_manager import get_hf_token

# Analysing which device is available
if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

question = "What does \"Crowd Control\" mean ?"
generator = pipeline("text-generation", model="avinot/LoLlama-3.2-1B-lora-3ep-v3-instruct", device=device, token=get_hf_token("read"))
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(f"Question: {question}")
print(f"{output["generated_text"]}")

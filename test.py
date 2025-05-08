# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import huggingface_hub
# from utils.token_manager import get_hf_token

# huggingface_hub.login(token=get_hf_token("read"))

# base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# # model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# model = PeftModel.from_pretrained(base_model, "avinot/Lollama3.2-1B-lora-3ep-v3")

# # Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# # Function to generate text
# def generate_text(prompt, max_length=200):
#     inputs = tokenizer(prompt, return_tensors="pt")
#     outputs = model.generate(**inputs, max_length=max_length, top_k=50, top_p=0.95)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # user_input = "In which champion class does the champion 'Thresh' belong to ?"
# user_input = "You are a League of Legends expert. Your role is to answer to the best of your abilities some questions about league of legends\nThresh first ability is..."
# response = generate_text(user_input.lower())
# print(response)

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-instruct")
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(base_model, "avinot/Lollama3.2-1B-lora-3ep-v3")
model = model.merge_and_unload()  # <<< Important line

prompt = "What is the role of the champion Thresh ?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=50, do_sample=False)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
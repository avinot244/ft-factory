from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import huggingface_hub
from utils.token_manager import get_hf_token

huggingface_hub.login(token=get_hf_token("read"))

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = PeftModel.from_pretrained(base_model, "avinot/Lollama3.2-1B-lora-50ep")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# Function to generate text
def generate_text(prompt, max_length=512):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length, top_k=50, top_p=0.95)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

user_input = "What should we do when playing Thresh support ?"
response = generate_text(user_input)
print(response)
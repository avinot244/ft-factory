import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.token_manager import get_hf_token

class PredictionHead(torch.nn.Module):
    def __init__(self, input_dim=2048, output_dim=2048, dtype=torch.bfloat16):
        super(PredictionHead, self).__init__()
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.backbone = AutoModelForCausalLM.from_pretrained(
            "avinot/LoLlama-3.2-1B-lora-3ep-v3-instruct", 
            token=get_hf_token("read"),
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ).to(self.device)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "avinot/LoLlama-3.2-1B-lora-3ep-v3-instruct", 
            token=get_hf_token("read")
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        self.proj = torch.nn.Linear(input_dim, output_dim, dtype=dtype).to(self.device)

    def forward(self, input_text : list[str]) -> torch.Tensor:
        # Ensure input_text is a list of strings
        if isinstance(input_text, str):
            input_text = [input_text]
        elif isinstance(input_text, (list, tuple)):
            # Convert any non-string elements to strings
            input_text = [str(item) for item in input_text]
        else:
            input_text = [str(input_text)]
        
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.backbone(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # tuple of (layer_count, B, T, D)
            last_k_layers = hidden_states[-10:]
            stacked = torch.stack(last_k_layers)  # (k, B, T, D)
            mean_k_layers = stacked.mean(dim=0)   # (B, T, D)

            attention_mask = inputs["attention_mask"].unsqueeze(-1).to(self.device)
            masked_mean = (mean_k_layers * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
        
        # Linear projection and L2 normalization
        projected = self.proj(masked_mean)
        normalized = torch.nn.functional.normalize(projected, p=2, dim=1)
            
        return normalized
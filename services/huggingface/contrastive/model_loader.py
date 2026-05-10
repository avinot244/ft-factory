import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.token_manager import get_hf_token


class PredictionHead(torch.nn.Module):
    def __init__(self, input_dim: int = 2048, proj_dim: int = 512, output_dim: int = 2048):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype   = torch.bfloat16

        self.backbone = AutoModelForCausalLM.from_pretrained(
            "avinot/LoLlama-3.2-1B-lora-3ep-v3-instruct",
            token=get_hf_token("read"),
            torch_dtype=self.dtype,
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "avinot/LoLlama-3.2-1B-lora-3ep-v3-instruct",
            token=get_hf_token("read"),
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.head = torch.nn.Sequential(
            torch.nn.Linear(input_dim, proj_dim, dtype=self.dtype).to(self.device),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(proj_dim, proj_dim, dtype=self.dtype).to(self.device),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(proj_dim, output_dim, dtype=self.dtype).to(self.device),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenise and encode a flat list of strings.
        Returns a normalised embedding tensor of shape (N, D).
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.backbone(**inputs, output_hidden_states=True)
            last_k  = torch.stack(outputs.hidden_states[-10:])   # (k, N, T, D)
            mean_hs = last_k.mean(dim=0)                          # (N, T, D)

            mask        = inputs["attention_mask"].unsqueeze(-1)  # (N, T, 1)
            masked_mean = (mean_hs * mask).sum(dim=1) / mask.sum(dim=1)  # (N, D)

        projected  = self.head(masked_mean)
        normalised = F.normalize(projected + masked_mean, p=2, dim=1)
        return normalised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, texts: list[str]) -> torch.Tensor:
        """
        Encode a batch of anchor / positive strings.
        Args:
            texts: list of B strings
        Returns:
            Tensor of shape (B, D)
        """
        if isinstance(texts, str):
            texts = [texts]
        return self._encode_texts(texts)

    def encode_negatives(self, negatives: list[list[str]]) -> torch.Tensor:
        """
        Encode a ragged batch of negatives into a dense padded tensor.

        Args:
            negatives: list of B lists, each containing K_i negative strings.
                       K_i may differ across rows.

        Returns:
            Tensor of shape (B, K_max, D) where rows with fewer than K_max
            negatives are zero-padded.  A boolean mask of shape (B, K_max)
            is returned as a second value (True = valid, False = padding).
        """
        B     = len(negatives)
        K_max = max(len(negs) for negs in negatives)

        # Encode all negatives in a single flat forward pass for efficiency
        flat_texts   = [text for negs in negatives for text in negs]
        flat_embeds  = self._encode_texts(flat_texts)          # (sum(K_i), D)
        D            = flat_embeds.size(-1)

        # Scatter back into (B, K_max, D) with zero padding
        packed = torch.zeros(B, K_max, D, dtype=flat_embeds.dtype, device=self.device)
        mask   = torch.zeros(B, K_max, dtype=torch.bool, device=self.device)
        cursor = 0
        for b, negs in enumerate(negatives):
            k = len(negs)
            packed[b, :k] = flat_embeds[cursor : cursor + k]
            mask[b, :k]   = True
            cursor += k

        return packed, mask   # (B, K_max, D), (B, K_max)
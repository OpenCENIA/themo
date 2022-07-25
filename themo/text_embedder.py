import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel

class TextEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    def __init__(
        self,
        version="openai/clip-vit-large-patch14", # clip-vit-base-patch32
        max_length=77
    ) -> None:
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.max_length = max_length
        self.freeze()

    def freeze(self) -> None:
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text) -> list:
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=False,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(next(iter(self.transformer.parameters())).device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.pooler_output # pooled (EOS token) states
        return z

    def encode(self, text) -> list:
        return self(text)  

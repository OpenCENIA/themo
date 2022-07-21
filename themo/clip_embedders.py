import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor

class FrozenCLIPTextEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for text (from huggingface)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length   # TODO: typical value?
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        pz = outputs.pooler_output # pooled (EOS token) states
        return z, pz

    def encode(self, text):
        return self(text)   

class FrozenCLIPVisionEmbedder(nn.Module):
    """Uses the CLIP transformer encoder for images (from huggingface)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):  # clip-vit-base-patch32
        super().__init__()
        self.processor = CLIPProcessor.from_pretrained(version)
        self.vision_model = CLIPVisionModel.from_pretrained(version)
        self.device = device
        self.freeze()

    def freeze(self):
        self.vision_model = self.vision_model.eval()
        #self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, image):
        batch_encoding = self.processor(images=image, return_tensors="pt").to(self.device)
        image_features = self.vision_model(**batch_encoding)

        z = image_features.last_hidden_state
        pz = image_features.pooler_output # pooled CLS states
        return z, pz

    def encode(self, text):
        return self(text)

import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers

import themo.data as data

__all__ = ["ThemoTextModel", "LitThemoTextModel", "BERT_MODEL_NAME"]

# should be kept in sync with themo.data.BERT_MODEL_NAME, this is bad design
# and we should look into this in the (near) future
BERT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"


class ThemoTextModel(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.transformer = transformers.BertModel.from_pretrained(
            BERT_MODEL_NAME, add_pooling_layer=False
        )
        transformer_width = self.transformer.config.hidden_size
        self.projection = nn.Linear(transformer_width, embed_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # only initialize projection i guess
        # taken from https://github.com/openai/CLIP/blob/f69a9bc217f6df9213628848b3f9b0b6fc542401/clip/model.py#L326 # noqa: E501
        for parameter in self.projection.parameters():
            nn.init.normal_(parameter, std=self.transformer.config.hidden_size**-0.5)

    # same signature as self.transformer
    def forward(self, *args, **kwargs) -> torch.Tensor:
        last_hidden_state = self.transformer(*args, **kwargs)[0]
        attn = kwargs["attention_mask"]
        pooled_hidden = (last_hidden_state * attn.unsqueeze(2)).sum(dim=1) / attn.sum(
            dim=1
        )[:, None]
        return self.projection(pooled_hidden)


class LitThemoTextModel(ThemoTextModel, pl.LightningModule):
    def __init__(self, embed_dim: int, learn_rate: float) -> None:
        super().__init__(embed_dim)

        self.save_hyperparameters()
        self.learn_rate = learn_rate

        self.loss = nn.MSELoss()

    def training_step(self, batch: data._Batch, batch_idx: int) -> torch.Tensor:
        input, target = batch
        output = self(**input)
        loss = self.loss(output, target)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: data._Batch, batch_idx: int) -> None:
        input, target = batch
        output = self(**input)
        loss = self.loss(output, target)
        self.log("val/loss", loss)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        # TODO: improve optimization. There are many options and it's not clear
        # what options are best, since we are not training from scratch but
        # finetuning a pretrained bert model
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learn_rate)
        return optimizer

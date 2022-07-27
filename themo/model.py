import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers


class ThemoTextModel(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        self.transformer = transformers.BertModel.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-uncased"
        )
        transformer_width = self.transformer.config.hidden_size
        self.ln_final = nn.LayerNorm(transformer_width)
        self.projection = (
            nn.Identity()
            if embed_dim == transformer_width
            else nn.Linear(transformer_width, embed_dim, bias=False)
        )

        self.reset_parameters()

    def reset_parameters(self):
        # only initialize projection i guess
        # taken from https://github.com/openai/CLIP/blob/f69a9bc217f6df9213628848b3f9b0b6fc542401/clip/model.py#L326
        for parameter in self.projection.parameters():
            nn.init.normal_(parameter, std=self.transformer.config.hidden_size**-0.5)

    # same signature as self.transformer
    def forward(self, *args, **kwargs):
        last_hidden_state, *_ = self.transformer(*args, **kwargs).values()
        normalized = self.ln_final(last_hidden_state)
        first_hidden = normalized[:, 0]
        return self.projection(first_hidden)


class LitThemoTextModel(ThemoTextModel, pl.LightningModule):
    def __init__(self, embed_dim: int, learn_rate: float) -> None:
        super(ThemoTextModel, self).__init__()
        super(LitThemoTextModel, self).__init__(embed_dim)

        self.save_hyperparameters()
        self.learn_rate = learn_rate

        self.loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        input, target = batch
        output = self(**input)
        loss = self.loss(output, target)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self(**input)
        loss = self.loss(output, target)
        self.log("val/loss", loss)

    def configure_optimizers(self):
        # TODO: improve optimization. There are many options and it's not clear
        # what options are best, since we are not training from scratch but
        # finetuning a pretrained bert model
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learn_rate)
        return optimizer

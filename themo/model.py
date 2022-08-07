import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
import typing_extensions as tpx

import themo.data as data

from ._constants import DEFAULT_TEXT_MODEL

__all__ = ["ThemoTextModel", "LitThemoTextModel"]


class AveragePooler(nn.Module):
    def forward(self, input: torch.Tensor, attention_mask: torch.Tensor):
        # input: B x L x H
        # attention_mask: B x L
        return torch.div(
            torch.sum(input * attention_mask[..., None], dim=1),
            torch.sum(attention_mask, dim=1, keepdim=True),
        )


class IndexPooler(nn.Module):
    def __init__(self, index: int) -> None:
        super().__init__()
        self.index = index

    def extra_repr(self) -> str:
        return str(self.index)

    def forward(self, input: torch.Tensor, *_) -> torch.Tensor:
        return input[:, self.index]


class ThemoTextModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        transformer_name: str = DEFAULT_TEXT_MODEL,
        use_last_layer_norm: bool = True,
        use_projection_bias: bool = False,
        init_strategy: tpx.Literal["clip", "mclip"] = "clip",
        pooling_type: tpx.Literal["first", "average"] = "first",
    ) -> None:
        super().__init__()
        self._init_strategy = init_strategy
        self.transformer = transformers.BertModel.from_pretrained(
            transformer_name, add_pooling_layer=False
        )
        transformer_width = self.transformer.config.hidden_size
        self.ln_final = (
            nn.LayerNorm(transformer_width) if use_last_layer_norm else nn.Identity()
        )
        self.projection = nn.Linear(
            transformer_width, embed_dim, bias=use_projection_bias
        )
        if pooling_type == "first":
            self.pooler = IndexPooler(0)
        elif pooling_type == "average":
            self.pooler = AveragePooler()
        else:
            raise Exception(
                "Mira la wea de tipo de pooling que me pides aweonao: "
                f"pooling_type={pooling_type}"
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # only initialize projection i guess
        # taken from https://github.com/openai/CLIP/blob/f69a9bc217f6df9213628848b3f9b0b6fc542401/clip/model.py#L326 # noqa: E501
        for name, parameter in self.projection.named_parameters():
            if self._init_strategy == "clip":
                nn.init.normal_(
                    parameter, std=self.transformer.config.hidden_size**-0.5
                )
            elif self._init_strategy == "mclip":
                if name == "weight":
                    nn.init.xavier_uniform_(parameter)
                elif name == "bias":
                    nn.init.zeros_(parameter)

    # same signature as self.transformer
    def forward(self, **kwargs) -> torch.Tensor:
        last_hidden_state = self.transformer(**kwargs).last_hidden_state
        normalized = self.ln_final(last_hidden_state)
        pooled = self.pooler(normalized, kwargs["attention_mask"])
        return self.projection(pooled)


class _SchedulerOpts(tpx.TypedDict):
    num_warmup_steps: int
    num_training_steps: int


class LitThemoTextModel(ThemoTextModel, pl.LightningModule):
    def __init__(
        self,
        embed_dim: int,
        transformer_name: str,
        use_last_layer_norm: bool,
        use_projection_bias: bool,
        init_strategy: tpx.Literal["clip", "mclip"],
        pooling_type: tpx.Literal["first", "average"],
        learn_rate: float,
        optimizer: tpx.Literal["adam", "adamw"],
        use_scheduler: bool,
        scheduler_opts: _SchedulerOpts,
    ) -> None:
        super().__init__(
            embed_dim,
            transformer_name,
            use_last_layer_norm,
            use_projection_bias,
            init_strategy,
            pooling_type,
        )

        assert scheduler_opts or not use_scheduler
        self.save_hyperparameters()

        self.learn_rate = learn_rate
        # hopefully they dont conflict with lightning's variables
        self._optimizer_name = optimizer
        self._use_scheduler = use_scheduler
        self._scheduler_opts = scheduler_opts

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

    def configure_optimizers(self) -> dict:
        # TODO: improve optimization. There are many options and it's not clear
        # what options are best, since we are not training from scratch but
        # finetuning a pretrained bert model
        optim_class = {
            "adam": torch.optim.Adam,
            "adamw": torch.optim.AdamW,
        }[self._optimizer_name]
        optimizer = optim_class(self.parameters(), lr=self.learn_rate)
        scheduler = (
            transformers.get_linear_schedule_with_warmup(
                optimizer, **self._scheduler_opts
            )
            if self._use_scheduler
            else torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": f"lr/{self._optimizer_name}",
            },
        }

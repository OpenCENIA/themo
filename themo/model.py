import typing_extensions as tpx
import torchmetrics
import tqdm
import typing as tp
import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers
import transformers.models.clip.modeling_clip as transformers_clip
from multilingual_clip import pt_multilingual_clip

import themo.data as data

__all__ = ["ThemoTextModel", "LitThemoTextModel", "BERT_MODEL_NAME"]

# NOTE: should be kept in sync with themo.data.BERT_MODEL_NAME, this is bad design
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


class ThemoModel(nn.Module):
    """This model as is won't work with HF transformers i think. We must find a
    better to make both compatible"""

    def __init__(
        self, text_model: ThemoTextModel, clip_model: transformers.CLIPModel
    ) -> None:
        """Hidden dimensions of text and clip model should match"""
        super().__init__()
        self.text_model = text_model
        self.clip_model = clip_model

    def forward(
        self,
        input_ids: tp.Optional[torch.LongTensor] = None,
        pixel_values: tp.Optional[torch.FloatTensor] = None,
        attention_mask: tp.Optional[torch.Tensor] = None,
        **_: tp.Any,
    ) -> transformers_clip.CLIPOutput:

        image_features = self.get_image_features(pixel_values)
        text_features = self.get_text_features(input_ids, attention_mask)

        # cosine similarity as logits
        logits_per_text = self.get_logits_per_text(image_features, text_features)
        logits_per_image = logits_per_text.t()

        return transformers_clip.CLIPOutput(
            logits_per_image=logits_per_image, logits_per_text=logits_per_text
        )

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.clip_model.vision_model(pixel_values=pixel_values)
        image_features = vision_outputs[1]
        image_features = self.clip_model.visual_projection(image_features)
        return image_features

    def get_text_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        text_features = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_features = self.clip_model.text_projection(text_features)
        return text_features

    def get_logits_per_text(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        return logits_per_text


class MultilingualCLIP(nn.Module):
    """Wrapper class for multilingual clip. This allows evaluating it with
    LitRetrievalWrapper"""

    MULTILINGUAL_MODEL_NAME = "M-CLIP/XLM-Roberta-Large-Vit-B-32"
    CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(self):
        """For now we evaluate only M-CLIP/XLM-Roberta-Large-Vit-B-32"""
        super().__init__()
        self.multilingual_clip = pt_multilingual_clip.MultilingualCLIP.from_pretrained(
            self.MULTILINGUAL_MODEL_NAME
        )
        self.clip_model = transformers.CLIPModel.from_pretrained(self.CLIP_MODEL_NAME)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.clip_model.get_image_features(pixel_values=pixel_values)

    def get_text_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mostly copied from mclip code. They tokenize inside the model's forward method, which is bad design in my opinion"""

        embs = self.multilingual_clip.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        embs = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(
            dim=1
        )[:, None]
        return self.multilingual_clip.LinearTransformation(embs)

    def get_logits_per_text(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        """Copied from the relevant part of transformers.CLIP forward"""
        # normalized features
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_text = torch.matmul(text_features, image_features.t()) * logit_scale
        return logits_per_text

    def get_logits_per_iamge(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        return self.get_logits_per_text(text_features, image_features).T


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


class RetrievalModel(tpx.Protocol):
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        ...

    def get_text_features(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        ...

    def get_logits_per_text(
        self, text_features: torch.Tensor, image_features: torch.Tensor
    ) -> torch.Tensor:
        ...

    def register_buffer(self, name: str, value: torch.Tensor, persistent: bool) -> None:
        ...


class LitRetrievalWrapper(pl.LightningModule):
    def __init__(self, model: RetrievalModel) -> None:
        super().__init__()

        self.model = model

        self.test_metrics = torchmetrics.MetricCollection(
            {
                "recall@01": torchmetrics.RetrievalRecall(
                    empty_target_action="error", k=1
                ),
                "recall@05": torchmetrics.RetrievalRecall(
                    empty_target_action="error", k=5
                ),
                "recall@10": torchmetrics.RetrievalRecall(
                    empty_target_action="error", k=10
                ),
            }
        )

    def on_test_start(self) -> None:
        """Computes the image features and saves them as a non-persistent
        buffer on the nn.Module"""

        # get a new test dataloader
        image_embedding_list = []
        dataloader = self.trainer.datamodule.test_dataloader()
        for batch in tqdm.tqdm(dataloader, desc="Computing image features"):
            pixel_values = batch["pixel_values"].to(self.device)
            image_embedding_list.append(self.model.get_image_features(pixel_values))

        all_image_embeddings = torch.vstack(image_embedding_list)
        self.model.register_buffer(
            "cached_image_embeddings", all_image_embeddings, persistent=False
        )

    def test_step(self, batch: transformers.BatchEncoding, batch_idx: int) -> None:
        text_embeds = self.model.get_text_features(
            batch["input_ids"], batch["attention_mask"]
        )
        logits_per_text = self.model.get_logits_per_text(
            text_embeds, self.model.cached_image_embeddings
        )
        original_batch_size = self.trainer.datamodule.hparams.batch_size
        current_batch_size, n_images = logits_per_text.shape

        # the target to be retrieved is just the image the caption came paired with, so
        # the index is the same index of the text in the whole dataset
        start_index = original_batch_size * batch_idx
        image_indices = torch.arange(
            start_index, start_index + current_batch_size, device=self.device
        )

        recall_target = (
            torch.arange(n_images, device=self.device) == image_indices[:, None]
        )
        recall_indexes = torch.arange(current_batch_size, device=self.device)[
            :, None
        ].repeat(1, n_images)
        self.log_dict(
            self.test_metrics(
                preds=logits_per_text, target=recall_target, indexes=recall_indexes
            )
        )

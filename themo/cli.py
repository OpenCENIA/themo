import torch
import click
import pytorch_lightning as pl
import themo


@click.command()
@click.option("--batch-size", default=128)
@click.option("--max-sequence-length", default=77)
@click.option("--learn-rate", default=5e-4)
def train(batch_size: int, max_sequence_length: int, learn_rate: float) -> None:
    datamodule = themo.WITParallelDataModule(
        datadir="data", batch_size=batch_size, max_sequence_length=max_sequence_length
    )
    model = themo.ThemoTextLitModel(
        embed_dim=themo.WITParallel.META.features_dim, learn_rate=learn_rate
    )
    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available else 0)
    trainer.fit(model, datamodule)

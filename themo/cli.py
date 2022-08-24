import click
import pytorch_lightning as pl
import torch
import transformers

import themo


def _configure_libs() -> None:
    # Transformers man, so f***ing verbose
    # transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
    pass


@click.command(context_settings=dict(show_default=True))
@click.option("--batch-size", default=32)
@click.option("--max-sequence-length", default=77)
@click.option("--learn-rate", default=5e-4)
def train(batch_size: int, max_sequence_length: int, learn_rate: float) -> str:
    """Trains themo with the given hparams, returns path to model with minimal
    train loss"""
    # Preliminary configs
    # ===================
    hparams = locals().copy()
    _configure_libs()
    # set seed?

    # Load model and datamodule
    # =========================
    clip_config = transformers.CLIPTextConfig.from_pretrained(
        themo.data.TARGET_FEATURES_MODEL
    )

    datamodule = themo.LitParallel(
        datadir="data", batch_size=batch_size, max_sequence_length=max_sequence_length
    )
    model = themo.LitThemoTextModel(
        embed_dim=clip_config.hidden_size, learn_rate=learn_rate
    )
    print("Data module ready!")
    # Trainer config
    # ==============
    checkpoint_callback_kwargs = {
        "monitor": "train/loss",
        "mode": "min",
    }
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename="epoch={epoch}_train-loss={train/loss:.3f}",
        auto_insert_metric_name=False,
        **checkpoint_callback_kwargs,
    )
    logger = pl.loggers.TensorBoardLogger(
        save_dir="logs",
        name="default",
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        gpus=-torch.cuda.is_available(),  # super cursed
        logger=logger,
        callbacks=[
            pl.callbacks.EarlyStopping(
                patience=5,
                **checkpoint_callback_kwargs,
            ),
            checkpoint_callback,
        ],
        max_epochs=1000,
    )
    trainer.fit(model, datamodule)
    logger.log_hyperparams(
        hparams, metrics=dict(best_loss=checkpoint_callback.best_model_score or -1)
    )
    return checkpoint_callback.best_model_path

import joblib
import pprint
import pathlib
import typing as tp
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


@click.group
def cli():
    pass


@cli.command(context_settings=dict(show_default=True))
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

    datamodule = themo.LitWITTranslated(
        datadir="data", batch_size=batch_size, max_sequence_length=max_sequence_length
    )
    model = themo.LitThemoTextModel(
        embed_dim=clip_config.hidden_size, learn_rate=learn_rate
    )
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


def _test_mclip(ctx: click.Context, param: click.Option, value: bool) -> None:
    # see docs here at
    # https://click.palletsprojects.com/en/8.1.x/options/#callbacks-and-eager-options
    if not value or ctx.resilient_parsing:
        return

    memory = joblib.Memory("data", verbose=0)

    @memory.cache
    def test_helper():
        """Allows result to be cached to disk"""
        model = themo.model.LitRetrievalWrapper(themo.model.MultilingualCLIP())
        datamodule = themo.data.LitXTD10(
            datadir="data",
            lang="es",
            batch_size=128,
            tokenizer_version=themo.model.MultilingualCLIP.MULTILINGUAL_MODEL_NAME,
            feature_extractor_version=themo.model.MultilingualCLIP.CLIP_MODEL_NAME,
        )
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=-1,
            logger=False,
        )
        return trainer.test(model, datamodule, verbose=False)

    pprint.pprint(test_helper())

    ctx.exit()


@cli.command()
@click.option(
    "--baseline", is_flag=True, callback=_test_mclip, expose_value=False, is_eager=True
)
@click.option(
    "--version-path",
    required=True,
    help=(
        "Path to the version dir. It has to be the version directory and not just the"
        " checkpoint, i.e something like `logs/.../version_x`."
    ),
)
@click.option("--batch-size", default=128)
def test(version_path: str, batch_size: int) -> None:

    # I'm loading the checkpoint from the version dir instead of the checkpoint
    # directly because we will need the info of the clip version that was used
    # to produce target features. That info is in the datamodule, so we will
    # require the full hparams.yaml file instead of just the ones saved in the
    # model.
    try:
        checkpoint_path = next(
            iter((pathlib.Path(version_path) / "checkpoints").iterdir())
        )
    except StopIteration:
        raise Exception(
            f"No checkpoint found in {pathlib.Path(version_path) / 'checkpoints'}"
        )

    text_model = themo.LitThemoTextModel.load_from_checkpoint(str(checkpoint_path))

    # For now use the default clip model
    clip_model = transformers.CLIPModel.from_pretrained(
        themo.data.TARGET_FEATURES_MODEL
    )
    themo_model = themo.model.ThemoModel(text_model, clip_model)
    model = themo.model.LitRetrievalWrapper(themo_model)
    datamodule = themo.data.LitXTD10(
        datadir="data",
        lang="es",
        batch_size=batch_size,
        tokenizer_version=themo.model.BERT_MODEL_NAME,
        feature_extractor_version=themo.data.TARGET_FEATURES_MODEL,
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=-1,
        logger=False,
    )
    results = trainer.test(model, datamodule, verbose=False)
    pprint.pprint(results)

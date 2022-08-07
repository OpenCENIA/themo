import datetime as dt
import json
import logging
import typing as tp

import click
import optuna
import pytorch_lightning as pl
import torch
import transformers

import themo


def train_single(
    batch_size: int,
    learn_rate: float,
    max_sequence_length: int,
    target_features_model: str,
    optimizer: str,
    use_scheduler: bool,
    scheduler_warmup_steps: int,
    scheduler_training_steps: int,
    use_last_layer_norm: bool,
    use_projection_bias: bool,
    init_strategy: str,
    pooling_type: str,
    fast_dev_run: bool = False,
    overfit_batches: int = 0,
    experiment_name: str = "default",
) -> float:
    """Trains themo with the given hparams, returns path to model with minimal
    train loss"""
    # Preliminary configs
    # ===================
    hparams = locals().copy()
    # set seed?

    # Load model and datamodule
    # =========================
    clip_config = transformers.CLIPTextConfig.from_pretrained(target_features_model)

    datamodule = themo.LitWITTranslated(
        datadir="data",
        batch_size=batch_size,
        max_sequence_length=max_sequence_length,
        clip_version=target_features_model,
    )
    model = themo.LitThemoTextModel(
        embed_dim=clip_config.hidden_size,
        transformer_name=themo.DEFAULT_TEXT_MODEL,
        use_last_layer_norm=use_last_layer_norm,
        use_projection_bias=use_projection_bias,
        init_strategy=init_strategy,
        pooling_type=pooling_type,
        learn_rate=learn_rate,
        optimizer=optimizer,
        use_scheduler=use_scheduler,
        scheduler_opts={
            "num_warmup_steps": scheduler_warmup_steps,
            "num_training_steps": scheduler_training_steps,
        },
    )
    hparams = {**hparams, **datamodule.hparams, **model.hparams}
    print("HPARAMS:", json.dumps(hparams, indent=4))
    # Trainer config
    # ==============
    checkpoint_callback_kwargs = {
        "monitor": "train/loss",
        "mode": "min",
    }
    early_stopping_callback = pl.callbacks.EarlyStopping(
        patience=3,
        **checkpoint_callback_kwargs,
    )
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     filename="epoch={epoch}_train-loss={train/loss:.3f}",
    #     auto_insert_metric_name=False,
    #     **checkpoint_callback_kwargs,
    # )
    logger = pl.loggers.TensorBoardLogger(
        save_dir="logs",
        name=experiment_name,
        default_hp_metric=False,
    )
    trainer = pl.Trainer(
        gpus=-torch.cuda.is_available(),  # super cursed
        logger=logger,
        callbacks=[
            early_stopping_callback,
            # checkpoint_callback,
            pl.callbacks.LearningRateMonitor(),
        ],
        max_epochs=1000,
        enable_checkpointing=False,
        fast_dev_run=fast_dev_run,
        overfit_batches=overfit_batches,
    )
    # Actually train
    # ==============
    trainer.fit(model, datamodule)
    logger.log_hyperparams(
        hparams,
        metrics={"hparams/best_loss": early_stopping_callback.best_score.item() or -1},
    )
    return early_stopping_callback.best_score.item()


@click.group()
def cli():
    pass


@cli.command()
@click.option("--n-trials", default=1)
@click.option("--experiment-name")
@click.option("--fast-dev-run", is_flag=True)
@click.option("--overfit-batches", type=int, default=0)
def optimize(
    n_trials: int,
    experiment_name: tp.Optional[str],
    fast_dev_run: bool,
    overfit_batches: int,
):
    transformers.logging.set_verbosity_error()
    logging.getLogger("pytorch_lightning").setLevel(logging.WARN)

    experiment_name = (
        experiment_name
        or f"experiment_{dt.datetime.now().isoformat(timespec='seconds')}"
    )

    def objective(trial):
        bools = [False, True]

        best_loss = train_single(
            trial.suggest_int("batch_size", 128, 256, step=16),
            trial.suggest_int("lr_mul", 1, 10)
            * 10 ** -trial.suggest_int("lr_exp", 4, 5),
            trial.suggest_int("max_sequence_length", 50, 100, step=5),
            trial.suggest_categorical(
                "target_features_model",
                [
                    themo.DEFAULT_FEATURES_MODEL,
                    "openai/clip-vit-base-patch16",
                    "openai/clip-vit-base-patch32",
                ],
            ),
            trial.suggest_categorical("optimizer", ["adam", "adamw"]),
            trial.suggest_categorical("use_scheduler", bools),
            trial.suggest_int("scheduler_warmup_steps", 500, 5000, step=500),
            trial.suggest_int("sched_train_steps_mul", 1, 10)
            * 10 ** trial.suggest_int("sched_train_steps_exp", 5, 7),
            trial.suggest_categorical("use_last_layer_norm", bools),
            trial.suggest_categorical("use_projection_bias", bools),
            trial.suggest_categorical("init_strategy", ["clip", "mclip"]),
            trial.suggest_categorical("pooling_type", ["first", "average"]),
            fast_dev_run=fast_dev_run,
            overfit_batches=overfit_batches,
            experiment_name=experiment_name,
        )

        return best_loss

    study = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(),
        study_name=experiment_name,
        direction="minimize",
    )
    study.optimize(objective, n_trials=n_trials)


@cli.command(context_settings=dict(show_default=True))
@click.option("--batch-size", required=True, type=int)
@click.option("--learn-rate", default=1e-5)
@click.option("--max-sequence-length", default=77)
@click.option(
    "--target-features-model",
    default=themo.DEFAULT_FEATURES_MODEL,
    type=click.Choice(
        [
            themo.DEFAULT_FEATURES_MODEL,  # ...-large-patch14
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
        ]
    ),
)
@click.option("--optimizer", type=click.Choice(["adam", "adamw"]), default="adam")
@click.option("--use-scheduler", is_flag=True, default=False)
@click.option("--scheduler-warmup-steps", default=10**3)
@click.option("--scheduler-training-steps", default=10**8)
@click.option("--use-last-layer-norm", is_flag=True, default=False)
@click.option("--use-projection-bias", is_flag=True, default=False)
@click.option("--init-strategy", type=click.Choice(["clip", "mclip"]), default="clip")
@click.option(
    "--pooling-type", type=click.Choice(["first", "average"]), default="first"
)
# debugging options
@click.option("--fast-dev-run", is_flag=True, default=False)
@click.option("--overfit-batches", default=0)
def train(**kwargs):
    # configure libs for single training
    # Transformers man, so f***ing verbose
    # transformers.logging.disable_progress_bar()
    transformers.logging.set_verbosity_error()
    logging.getLogger("pytorch_lightning").setLevel(logging.INFO)

    train_single(**kwargs)

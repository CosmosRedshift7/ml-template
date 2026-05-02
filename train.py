import argparse

from aim import Image
from aim.pytorch_lightning import AimLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from model import LightningModel, LinearRegressionData
from utils import ensure_dir, load_config, make_config_parser, save_test_fit_plot


def parse_args() -> argparse.Namespace:
    parser = make_config_parser("Train a Lightning model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg["seed"], workers=True)

    ensure_dir("local")
    ensure_dir(cfg["aim"]["repo"])

    datamodule = LinearRegressionData(**cfg["data"])
    model = LightningModel(**cfg["model"], **cfg["optim"])

    logger = AimLogger(
        repo=cfg["aim"]["repo"],
        experiment=cfg["aim"]["experiment_name"],
    )
    logger.log_hyperparams(cfg)

    checkpoint = ModelCheckpoint(**cfg["checkpoint"])
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    trainer = Trainer(
        **cfg["trainer"],
        logger=logger,
        callbacks=[checkpoint, lr_monitor],
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    fig_path = save_test_fit_plot(
        model,
        datamodule,
        "local/figures/train_fit_on_test_dataset.png",
    )

    logger.experiment.track(
        Image(str(fig_path)),
        name="train_fit_on_test_dataset",
        context={"type": "figure"},
    )

    print(f"Best checkpoint saved to: {checkpoint.best_model_path}")
    print(f"Test fit plot saved to: {fig_path}")


if __name__ == "__main__":
    main()

import argparse

import mlflow
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from model import LightningModel, LinearRegressionData
from utils import ensure_dir, load_config, make_config_parser


def parse_args() -> argparse.Namespace:
    parser = make_config_parser("Train a Lightning model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg["seed"], workers=True)

    ensure_dir("local")
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    datamodule = LinearRegressionData(**cfg["data"])
    model = LightningModel(**cfg["model"], **cfg["optim"])

    logger = MLFlowLogger(
        experiment_name=cfg["mlflow"]["experiment_name"],
        tracking_uri=tracking_uri,
        log_model=False,
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
    trainer.test(model, datamodule=datamodule, ckpt_path="best")

    print(f"Best checkpoint saved to: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()

import argparse

import mlflow
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from model import LightningModel, LinearRegressionData
from utils import load_config, make_config_parser


def parse_args() -> argparse.Namespace:
    parser = make_config_parser("Evaluate a trained Lightning model.")
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    tracking_uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    datamodule = LinearRegressionData(**cfg["data"])
    model = LightningModel(**cfg["model"], **cfg["optim"])

    logger = MLFlowLogger(
        experiment_name=cfg["mlflow"]["experiment_name"],
        tracking_uri=tracking_uri,
        log_model=False,
    )

    trainer = Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        logger=logger,
    )

    trainer.test(model, datamodule=datamodule, ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()

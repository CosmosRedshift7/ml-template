import argparse

import torch
from aim.pytorch_lightning import AimLogger
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from callbacks import AimPlotCallback
from model import LightningModel, LinearRegressionData
from utils import ensure_dir, load_config, make_config_parser


def parse_args() -> argparse.Namespace:
    parser = make_config_parser("Train a Lightning model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    precision = cfg.get("torch", {}).get("float32_matmul_precision", "highest")
    torch.set_float32_matmul_precision(precision)

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
    plot_callback = AimPlotCallback(
        save_dir="local/figures",
        every_n_epochs=None,
        track_on_train_end=False,
        track_first_middle_last=True,
    )

    trainer = Trainer(
        **cfg["trainer"],
        logger=logger,
        callbacks=[checkpoint, lr_monitor, plot_callback],
        deterministic=True,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)

    print(f"Best checkpoint saved to: {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()

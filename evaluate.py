import argparse

from aim.pytorch_lightning import AimLogger
from lightning.pytorch import Trainer

from callbacks import AimPlotCallback
from model import LightningModel, LinearRegressionData
from utils import ensure_dir, load_config, make_config_parser


def parse_args() -> argparse.Namespace:
    parser = make_config_parser("Evaluate a trained Lightning model.")
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    ensure_dir("local")
    ensure_dir(cfg["aim"]["repo"])

    ckpt_path = args.ckpt or cfg["evaluate"]["ckpt_path"]

    datamodule = LinearRegressionData(**cfg["data"])
    model = LightningModel.load_from_checkpoint(ckpt_path)

    logger = AimLogger(
        repo=cfg["aim"]["repo"],
        experiment=cfg["aim"]["experiment_name"],
    )
    logger.log_hyperparams(cfg)

    plot_callback = AimPlotCallback(
        save_dir="local/figures",
        track_on_test_end=True,
    )

    trainer = Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        logger=logger,
        callbacks=[plot_callback],
    )

    trainer.test(model, datamodule=datamodule)

    print(f"Evaluated checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

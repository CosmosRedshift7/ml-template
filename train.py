import argparse

import mlflow
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

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
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    datamodule = LinearRegressionData(**cfg["data"])
    model = LightningModel(**cfg["model"], **cfg["optim"])

    experiment_name = cfg["mlflow"]["experiment_name"]
    artifact_location = cfg["mlflow"]["artifact_location"]

    mlflow.set_tracking_uri(tracking_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )

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
    trainer.test(model, datamodule=datamodule)

    fig_path = save_test_fit_plot(
        model,
        datamodule,
        f"local/figures/train_fit_on_test_dataset_{logger.run_id}.png",
    )

    logger.experiment.log_artifact(
        logger.run_id,
        str(fig_path),
        artifact_path="figures",
    )

    print(f"Best checkpoint saved to: {checkpoint.best_model_path}")
    print(f"Test fit plot saved to: {fig_path}")


if __name__ == "__main__":
    main()

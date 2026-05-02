import argparse

import mlflow
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import MLFlowLogger

from model import LightningModel, LinearRegressionData
from utils import ensure_dir, load_config, make_config_parser, save_test_fit_plot


def parse_args() -> argparse.Namespace:
    parser = make_config_parser("Evaluate a trained Lightning model.")
    parser.add_argument("--ckpt", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    ensure_dir("local")
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    ckpt_path = args.ckpt or cfg["evaluate"]["ckpt_path"]

    datamodule = LinearRegressionData(**cfg["data"])
    model = LightningModel.load_from_checkpoint(ckpt_path)

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

    trainer = Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        logger=logger,
    )

    trainer.test(model, datamodule=datamodule)

    evaluated_model = LightningModel.load_from_checkpoint(ckpt_path)
    evaluated_model.to(trainer.strategy.root_device)

    fig_path = save_test_fit_plot(
        evaluated_model,
        datamodule,
        f"local/figures/evaluate_fit_on_test_dataset_{logger.run_id}.png",
    )

    logger.experiment.log_artifact(
        logger.run_id,
        str(fig_path),
        artifact_path="figures",
    )

    print(f"Evaluated checkpoint: {ckpt_path}")
    print(f"Test fit plot saved to: {fig_path}")


if __name__ == "__main__":
    main()

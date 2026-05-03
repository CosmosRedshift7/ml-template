from pathlib import Path

import torch
from aim import Image
from aim.pytorch_lightning import AimLogger
from lightning.pytorch import Callback, Trainer

from model import LightningModel
from utils import save_plot


class AimPlotCallback(Callback):
    """Save predicted-vs-true fit plots locally and track them in Aim."""

    def __init__(
        self,
        save_dir: str = "local/figures",
        every_n_epochs: int | None = None,
        track_on_train_end: bool = False,
        track_first_middle_last: bool = False,
        track_on_test_end: bool = False,
        train_series_name: str = "fit_on_test_dataset",
        test_series_name: str = "fit_on_test_dataset",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.every_n_epochs = every_n_epochs
        self.track_on_train_end = track_on_train_end
        self.track_first_middle_last = track_first_middle_last
        self.track_on_test_end = track_on_test_end
        self.train_series_name = train_series_name
        self.test_series_name = test_series_name

    def _collect_test_predictions(
        self,
        model: LightningModel,
        datamodule,
    ) -> tuple:
        datamodule.setup("test")
        test_loader = datamodule.test_dataloader()

        model.eval()
        device = model.device

        y_true_list = []
        y_pred_list = []

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                pred = model(x).cpu()

                y_true_list.append(y.cpu())
                y_pred_list.append(pred)

        y_true = torch.cat(y_true_list).squeeze().numpy()
        y_pred = torch.cat(y_pred_list).squeeze().numpy()

        return y_true, y_pred

    def _should_track_epoch(self, trainer: Trainer, epoch: int) -> bool:
        max_epochs = trainer.max_epochs

        if self.every_n_epochs is not None and epoch % self.every_n_epochs == 0:
            return True

        if self.track_first_middle_last and max_epochs is not None and max_epochs > 0:
            middle_epoch = max(1, max_epochs // 2)
            selected_epochs = {1, middle_epoch, max_epochs}
            return epoch in selected_epochs

        return False

    def _save_and_track_plot(
        self,
        trainer: Trainer,
        pl_module: LightningModel,
        series_name: str,
        title: str,
        epoch: int | None = None,
    ) -> None:
        datamodule = trainer.datamodule
        logger = trainer.logger

        if datamodule is None:
            return

        if not isinstance(logger, AimLogger):
            return

        self.save_dir.mkdir(parents=True, exist_ok=True)

        if epoch is None:
            fig_path = self.save_dir / f"{series_name}.png"
        else:
            fig_path = self.save_dir / f"{series_name}_epoch_{epoch}.png"

        y_true, y_pred = self._collect_test_predictions(pl_module, datamodule)

        save_plot(
            y_true,
            y_pred,
            fig_path,
            title=title,
        )

        track_kwargs = {
            "value": Image(str(fig_path)),
            "name": series_name,
            "context": {"type": "figure"},
        }
        if epoch is not None:
            track_kwargs["step"] = epoch

        logger.experiment.track(**track_kwargs)

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModel) -> None:
        epoch = trainer.current_epoch + 1

        if self._should_track_epoch(trainer, epoch):
            self._save_and_track_plot(
                trainer,
                pl_module,
                series_name=self.train_series_name,
                title=f"Epoch {epoch}",
                epoch=epoch,
            )

    def on_train_end(self, trainer: Trainer, pl_module: LightningModel) -> None:
        if not self.track_on_train_end:
            return

        epoch = trainer.current_epoch

        if self._should_track_epoch(trainer, epoch):
            return

        self._save_and_track_plot(
            trainer,
            pl_module,
            series_name=self.train_series_name,
            title=f"Epoch {epoch}",
            epoch=epoch,
        )

    def on_test_end(self, trainer: Trainer, pl_module: LightningModel) -> None:
        if not self.track_on_test_end:
            return

        self._save_and_track_plot(
            trainer,
            pl_module,
            series_name=self.test_series_name,
            title="Evaluation fit",
            epoch=None,
        )

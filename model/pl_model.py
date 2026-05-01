import torch
from lightning.pytorch import LightningModule
from torch import Tensor

from model.loss import mse_loss
from model.model import FCNet


class LightningModel(LightningModule):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()

        self.lr = lr
        self.weight_decay = weight_decay

        self.model = FCNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        x, y = batch
        pred = self(x)
        loss = mse_loss(pred, y)

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

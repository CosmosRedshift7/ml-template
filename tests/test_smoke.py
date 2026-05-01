import torch

from model import LightningModel, LinearRegressionData


def test_model_forward_shape() -> None:
    model = LightningModel(input_dim=16, hidden_dim=32, output_dim=1)
    x = torch.randn(8, 16)
    y = model(x)
    assert y.shape == (8, 1)


def test_datamodule_batches() -> None:
    dm = LinearRegressionData(n_train=16, n_val=8, n_test=8, batch_size=4)
    dm.setup()
    x, y = next(iter(dm.train_dataloader()))
    assert x.shape == (4, 16)
    assert y.shape == (4, 1)

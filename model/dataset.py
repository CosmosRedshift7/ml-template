import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class LinearRegressionData(LightningDataModule):
    """DataModule for a toy linear regression problem.

    The module generates random input vectors x sampled from a standard normal
    distribution:

        x ~ N(0, 1),    x.shape = (n_samples, input_dim)

    It then generates scalar targets y using a fixed randomly sampled weight
    vector w and additive Gaussian noise:

        w ~ N(0, 1)
        noise ~ N(0, noise_std^2)
        y = x @ w + noise

    The outputs are PyTorch DataLoaders for the train, validation,
    and test splits.

    Each batch returned by the DataLoaders has the form:

        x_batch.shape = (batch_size, input_dim)
        y_batch.shape = (batch_size, 1)
    """

    def __init__(
        self,
        n_train: int = 4096,
        n_val: int = 1024,
        n_test: int = 1024,
        input_dim: int = 16,
        batch_size: int = 128,
        num_workers: int = 0,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()

        self.n_train = n_train
        self.n_val = n_val
        self.n_test = n_test
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.noise_std = noise_std

        self.save_hyperparameters()

    def setup(self, stage: str | None = None) -> None:
        generator = torch.Generator().manual_seed(12345)

        true_w = torch.randn(self.input_dim, 1, generator=generator)

        def make_split(n_samples: int) -> TensorDataset:
            x = torch.randn(n_samples, self.input_dim, generator=generator)
            noise = self.noise_std * torch.randn(n_samples, 1, generator=generator)
            y = x @ true_w + noise
            return TensorDataset(x, y)

        self.train_dataset = make_split(self.n_train)
        self.val_dataset = make_split(self.n_val)
        self.test_dataset = make_split(self.n_test)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

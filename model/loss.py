import torch
from torch import Tensor


def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return torch.mean((pred - target) ** 2)

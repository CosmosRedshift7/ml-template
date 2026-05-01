import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import yaml

from model import LightningModel, LinearRegressionData


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_config_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    return parser


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_test_fit_plot(
    model: LightningModel,
    datamodule: LinearRegressionData,
    save_path: str | Path,
) -> Path:
    """Save a predicted-vs-true plot on the test set.

    For multidimensional inputs, the most natural visualization is a scatter
    plot of predicted targets versus true targets. The dashed diagonal line
    shows the ideal fit y_pred = y_true.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    datamodule.setup("test")
    test_loader = datamodule.test_dataloader()

    model.eval()

    y_true_list = []
    y_pred_list = []

    device = model.device

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            pred = model(x).cpu()

            y_true_list.append(y.cpu())
            y_pred_list.append(pred)

    y_true = torch.cat(y_true_list).squeeze().numpy()
    y_pred = torch.cat(y_pred_list).squeeze().numpy()

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True target")
    plt.ylabel("Predicted target")
    plt.title("Test fit: predicted vs true")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path

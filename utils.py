import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from numpy.typing import ArrayLike


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


def save_plot(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    save_path: str | Path,
    title: str = "Predicted vs true",
) -> Path:
    """Save a predicted-vs-true scatter plot.

    The dashed diagonal line shows the ideal fit y_pred = y_true.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).squeeze()
    y_pred = np.asarray(y_pred).squeeze()

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.figure(figsize=(3, 3))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    return save_path

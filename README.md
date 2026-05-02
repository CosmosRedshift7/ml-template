# ml-template

Minimal machine learning research template using [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/), [MLflow](https://mlflow.org/), and [Pixi](https://pixi.sh/).

This repository is designed to be a clean starting point for ML experiments, research code, and small-to-medium prototype projects. The goal is to keep the project **reproducible**, **easy to understand**, and **easy to copy into a new project**.

Reproducibility is handled through Pixi environments and the generated `pixi.lock` file. After dependencies are resolved once, the lock file records the exact package versions, so another machine can recreate the same environment instead of playing the traditional “works on my machine” academic sport. Pixi is designed around reproducible, cross-platform environments and one-command task execution. [PyTorch](https://pytorch.org/projects/pytorch/) provides the core deep learning framework, [Lightning](https://lightning.ai/docs/pytorch/latest/) organizes training/evaluation code, and [MLflow](https://mlflow.org/) tracks metrics, parameters, and artifacts. :contentReference[oaicite:0]{index=0}

## Why use this template?

This template gives you a practical baseline for ML projects without forcing a heavy framework on top of your code.

Main benefits:

- **Reproducible environments** with Pixi and `pixi.lock`.
- **Simple training loop** using PyTorch Lightning.
- **Experiment tracking** with local MLflow.
- **Config-driven experiments** through `configs/default.yaml`.
- **Clean project structure** separating data, model, loss, training, evaluation, and utilities.
- **Local outputs kept out of git** through the ignored `local/` directory.
- **Ready-to-run example** using a toy linear regression dataset.
- **Smoke tests included** so you can quickly check that the template still works.
- **Useful Pixi tasks** for training, evaluation, MLflow UI, formatting, linting, testing, and cleanup.

This is intentionally small. It is meant to be copied, modified, and extended.

## Structure

```text
.
├── train.py
├── evaluate.py
├── utils.py
├── pyproject.toml
├── pixi.lock
├── README.md
├── LICENSE
├── .gitignore
├── configs/
│   └── default.yaml
├── local/              # ignored by git: data, runs, checkpoints, figures
├── model/
│   ├── __init__.py
│   ├── dataset.py
│   ├── loss.py
│   ├── model.py
│   └── pl_model.py
└── tests/
    └── test_smoke.py
```

## Setup

Install Pixi first if you do not already have it:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

Then install the project environment:

```bash
pixi install
```

This creates a local Pixi environment using the dependencies specified in `pyproject.toml` and locked in `pixi.lock`.

## Train

Run training with:

```bash
pixi run train
```

or manually:

```bash
pixi run python train.py --config configs/default.yaml
```

Training will:

- load configuration from `configs/default.yaml`,
- train a small fully connected model,
- log metrics and hyperparameters to MLflow,
- save checkpoints under `local/checkpoints/`,
- save a test fit plot under `local/figures/`,
- log the plot as an MLflow artifact.

## Evaluate from a checkpoint

Evaluate the checkpoint specified in `configs/default.yaml`:

```bash
pixi run evaluate
```

By default, this evaluates:

```text
local/checkpoints/best.ckpt
```

To evaluate a different checkpoint:

```bash
pixi run python evaluate.py --config configs/default.yaml --ckpt path/to/checkpoint.ckpt
```

Evaluation logs test metrics and saves a predicted-vs-true fit plot.

## Open MLflow UI

Start the local MLflow UI:

```bash
pixi run mlflow-ui
```

Then open:

```text
http://127.0.0.1:5000
```

In the MLflow UI, select the `ml-template` experiment. You should see runs with logged parameters, metrics such as `train/loss`, `val/loss`, and `test/loss`, and generated figure artifacts.

## Configuration

The main configuration file is:

```text
configs/default.yaml
```

It controls:

- random seed,
- dataset sizes,
- input dimension,
- batch size,
- model dimensions,
- optimizer settings,
- trainer settings,
- MLflow tracking URI,
- checkpoint path,
- evaluation checkpoint path.

Example:

```yaml
trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
  log_every_n_steps: 10
```

## Local outputs

Generated files are stored under `local/`, which is ignored by git.

Typical local outputs:

```text
local/mlflow.db
local/checkpoints/
local/figures/
```

This keeps the repository clean while allowing experiments, checkpoints, plots, and temporary files to stay available locally.

## Cleaning local outputs

Clean only MLflow runs and experiment metadata:

```bash
pixi run clean-runs
```

Clean only model checkpoints:

```bash
pixi run clean-checkpoints
```

Clean only generated figures:

```bash
pixi run clean-figures
```

Clean everything generated locally:

```bash
pixi run clean-all
```

The cleanup tasks remove these files/directories:

```text
local/mlflow.db
local/mlruns/
mlruns/
local/checkpoints/
local/figures/
```

## Format, lint, and test

Format code:

```bash
pixi run format
```

Lint code:

```bash
pixi run lint
```

Run tests:

```bash
pixi run pytest
```

## Recommended workflow

A typical workflow is:

```bash
pixi install
pixi run train
pixi run mlflow-ui
```

## Extending the template

Common next steps:

- Replace `LinearRegressionData` in `model/dataset.py` with your own data module.
- Replace `FCNet` in `model/model.py` with your own neural network model.
- Modify `mse_loss` in `model/loss.py` or add new loss functions.
- Add more configuration files under `configs/`.
- Add project-specific metrics, plots, callbacks, or MLflow artifacts.
- Add real unit tests under `tests/`.

## Notes

- `local/` is intentionally ignored by git.
- Keep raw data, generated data, MLflow runs, checkpoints, and figures under `local/`.
- Commit `pixi.lock` for reproducible environments.
- The default example trains a tiny fully connected model on a synthetic linear regression dataset.
- The template uses local MLflow tracking through SQLite by default.
- Model checkpoints are saved locally rather than uploaded to a cloud registry.

## License

This project is licensed under the MIT License.

# ml-template

Minimal ML research template using **PyTorch**, **PyTorch Lightning**, **MLflow**, and **Pixi**.

The goal is to keep the project boring, reproducible, and easy to copy into new research projects.

## Structure

```text
.
├── train.py
├── test.py
├── utils.py
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   └── default.yaml
├── local/              # ignored by git: data, runs, temporary files
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

```bash
pixi install
```

## Train

```bash
pixi run train
```

or manually:

```bash
pixi run python train.py --config configs/default.yaml
```

Training logs are written to MLflow in `local/mlruns`.

## Test / evaluate from a checkpoint

```bash
pixi run test
```

or:

```bash
pixi run python test.py --config configs/default.yaml --ckpt local/checkpoints/best.ckpt
```

## Open MLflow UI

```bash
pixi run mlflow-ui
```

Then open:

```text
http://127.0.0.1:5000
```

## Cleaning local outputs

The template stores generated experiment files under `local/`, which is ignored by git.

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

```bash
pixi run format
pixi run lint
pixi run pytest
```

## Notes

- `local/` is intentionally ignored by git.
- Keep raw data, generated data, MLflow runs, and checkpoints under `local/`.
- Commit `pixi.lock` once generated for reproducible environments.
- The default example trains a tiny MLP on a synthetic regression dataset.

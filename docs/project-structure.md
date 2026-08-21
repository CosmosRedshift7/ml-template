# Project structure

The template uses a small but expandable structure.

```text
.
├── train.py
├── evaluate.py
├── callbacks.py
├── clean.py
├── utils.py
├── pyproject.toml
├── pixi.lock
├── README.md
├── LICENSE
├── .gitignore
├── configs/
│   └── default.yaml
├── local/
├── model/
│   ├── __init__.py
│   ├── dataset.py
│   ├── loss.py
│   ├── model.py
│   └── pl_model.py
└── tests/
    └── test_smoke.py
```

## Top-level files

| File | Purpose |
| --- | --- |
| `train.py` | Training entry point |
| `evaluate.py` | Evaluation entry point |
| `callbacks.py` | Aim figure logging callback |
| `clean.py` | Cross-platform cleanup helper for generated outputs |
| `utils.py` | Shared helper functions |
| `pyproject.toml` | Project metadata, dependencies, Pixi tasks |
| `pixi.lock` | Locked dependency versions |
| `.gitignore` | Keeps generated files out of git |

## `configs/`

Contains YAML configuration files.

Default file:

```text
configs/default.yaml
```

Use this folder for experiment configs.

## `model/`

Contains the ML code:

| File | Purpose |
| --- | --- |
| `dataset.py` | Data module |
| `model.py` | Neural network architecture |
| `loss.py` | Loss functions |
| `pl_model.py` | PyTorch Lightning module |

## `tests/`

Contains smoke tests and future unit tests.

Default file:

```text
tests/test_smoke.py
```

## `local/`

This folder is ignored by git and is intended for generated local artifacts:

```text
local/aim/
local/checkpoints/
local/figures/
```

Keep raw data, generated data, checkpoints, runs, and temporary outputs under `local/`.

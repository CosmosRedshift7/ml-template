# ml-template

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.6-purple)](https://lightning.ai/docs/pytorch/latest/)
[![Aim](https://img.shields.io/badge/Aim-experiment%20tracking-111111)](https://aimstack.io/)
[![Pixi](https://img.shields.io/badge/Pixi-reproducible%20envs-f0b90b)](https://pixi.sh/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A lightweight, reproducible PyTorch Lightning template with local Aim experiment tracking and Pixi environments.

Minimal machine learning research template using [PyTorch](https://pytorch.org/), [PyTorch Lightning](https://lightning.ai/docs/pytorch/latest/), [Aim](https://aimstack.io/), and [Pixi](https://pixi.sh/).

This repository is designed to be a clean starting point for ML experiments, research code, and small-to-medium prototype projects. The goal is to keep the project **reproducible**, **easy to understand**, and **easy to copy into a new project**.

Reproducibility is handled through Pixi environments and the generated `pixi.lock` file. After dependencies are resolved once, the lock file records the exact package versions, so another machine can recreate the same environment instead of playing the traditional "works on my machine" academic sport. Pixi provides reproducible environments and one-command task execution, PyTorch provides the core deep learning framework, Lightning organizes training/evaluation code, and Aim tracks metrics, parameters, and figures locally through a web UI.

## Quick start

Clone the repository:

```bash
git clone https://github.com/CosmosRedshift7/ml-template.git
cd ml-template
```

Install the environment, train the example model, and start the Aim UI:

```bash
pixi install
pixi run train
pixi run aim-ui
```

Then open:

```text
http://127.0.0.1:43800
```

> [!TIP]
> In the Aim UI, open the `ml-template` experiment to view runs, metrics, hyperparameters, and tracked figures.

## What you get

| Feature                  | Included                               |
| ------------------------ | -------------------------------------- |
| Reproducible environment | Pixi + `pixi.lock`                     |
| Training framework       | PyTorch Lightning                      |
| Multi-GPU training       | Configurable through Lightning Trainer |
| Experiment tracking      | Local Aim tracking                     |
| Configuration            | YAML config in `configs/default.yaml`  |
| Checkpointing            | Lightning `ModelCheckpoint`            |
| Evaluation               | Separate `evaluate.py` entry point     |
| Plot tracking            | Aim callback for plots                 |
| Tests                    | Pytest smoke tests                     |
| Code quality             | Ruff formatting and linting            |
| Local cleanup            | Pixi cleanup tasks                     |

## Why use this template?

Main benefits:

- **Reproducible environments** with Pixi and `pixi.lock`.
- **Simple training loop** using PyTorch Lightning.
- **Easy multi-GPU training** through Lightning Trainer settings such as `accelerator`, `devices`, and `strategy`.
- **Local experiment tracking** with Aim.
- **Config-driven experiments** through `configs/default.yaml`.
- **Clean project structure** separating data, model, loss, training, evaluation, callbacks, and utilities.
- **Local outputs kept out of git** through the ignored `local/` directory.
- **Ready-to-run example** using a toy linear regression dataset.
- **Smoke tests included** so you can quickly check that the template still works.
- **Useful Pixi tasks** for training, evaluation, Aim UI, formatting, linting, testing, and cleanup.
- **Reusable callback pattern** for logging figures during training and evaluation.

## Structure

```text
.
├── train.py
├── evaluate.py
├── callbacks.py
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

## Setup

Install Pixi first if you do not already have it.

Linux & macOS:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Windows:

[Download installer](https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.msi)

or install from PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

> [!IMPORTANT]
> 🔥 **Restart your terminal or shell after installing Pixi.**
>
> This makes the `pixi` command available in your shell.

Then install the project environment:

```bash
pixi install
```

This creates a local Pixi environment using the dependencies specified in `pyproject.toml` and locked in `pixi.lock`.

> [!TIP]
> Commit `pixi.lock` to make the environment reproducible across machines.

> [!NOTE]
> The default environment uses CPU PyTorch. For CUDA-enabled training, see the [GPU training](#gpu-training) section.

## Managing Pixi environments

Activate the project environment in your terminal:

```bash
pixi shell
```

This lets you run commands such as `python`, `pytest`, or `ruff` directly inside the Pixi environment.

> [!TIP]
> Use `pixi shell` when you want your terminal or editor to use the project environment interactively.

To rebuild the Pixi environment from the lock file:

```bash
rm -rf .pixi
pixi install
```

To fully resolve dependencies again and regenerate the lock file:

```bash
rm -rf .pixi pixi.lock
pixi install
```

> [!WARNING]
> Deleting `.pixi/` removes the local environment. Deleting `pixi.lock` forces Pixi to resolve package versions again, which may produce a different environment.

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
- track metrics and hyperparameters with Aim,
- save checkpoints under `local/checkpoints/`,
- save predicted-vs-true fit plots under `local/figures/`,
- track selected fit plots in Aim.

By default, the training callback tracks predicted-vs-true plots for selected epochs, such as the first, middle, and final epoch.

> [!NOTE]
> Training outputs are saved under `local/`, which is ignored by git.

## GPU training

PyTorch Lightning makes it easy to use the same training script on CPU, single-GPU, or multi-GPU machines.

This template defines separate Pixi environments for CPU and GPU usage:

```text
cpu      # CPU PyTorch environment
gpu      # CUDA-enabled PyTorch environment
default  # uses the CPU environment by default
```

The default environment uses CPU PyTorch, so normal training works with:

```bash
pixi run train
```

or explicitly:

```bash
pixi run -e cpu train
```

For CUDA-enabled PyTorch, install the GPU environment:

```bash
pixi install -e gpu
```

Check that PyTorch can see CUDA:

```bash
pixi run -e gpu python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'
```

Expected output should look similar to:

```text
True
12.9
```

With the default trainer settings in `configs/default.yaml`,

```yaml
trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
```

running the GPU environment will automatically use a GPU if one is available:

```bash
pixi run -e gpu train
```

For explicit GPU control, edit `configs/default.yaml`:

```yaml
# single GPU
trainer:
  accelerator: gpu
  devices: 1
```

```yaml
# two GPUs with distributed data parallel training
trainer:
  accelerator: gpu
  devices: 2
  strategy: ddp
```

```yaml
# all available GPUs
trainer:
  accelerator: gpu
  devices: auto
  strategy: ddp
```

> [!IMPORTANT]
> GPU training requires NVIDIA GPUs, a compatible NVIDIA driver, and the CUDA-enabled Pixi environment. The CPU environment is kept as the default because it works on most machines.

## Evaluate from a checkpoint

Evaluate the checkpoint specified in `configs/default.yaml`:

```bash
pixi run evaluate
```

By default, this evaluates:

```text
local/checkpoints/best.ckpt
```

> [!IMPORTANT]
> Run `pixi run train` before `pixi run evaluate`, unless you already have a checkpoint at `local/checkpoints/best.ckpt`.

To evaluate a different checkpoint:

```bash
pixi run python evaluate.py --config configs/default.yaml --ckpt path/to/checkpoint.ckpt
```

Evaluation logs test metrics and tracks a predicted-vs-true fit plot in Aim.

## Open Aim UI

Start the local Aim UI:

```bash
pixi run aim-ui
```

Then open:

```text
http://127.0.0.1:43800
```

In the Aim UI, open the `ml-template` experiment. You should see runs with tracked parameters, metrics such as `train/loss`, `val/loss`, and `test/loss`, and generated image sequences such as predicted-vs-true fit plots.

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
- Aim repository path,
- checkpoint path,
- evaluation checkpoint path.

## Local outputs

Generated files are stored under `local/`, which is ignored by git.

Typical local outputs:

```text
local/aim/
local/checkpoints/
local/figures/
```

This keeps the repository clean while allowing experiments, checkpoints, plots, and temporary files to stay available locally.

## Cleaning local outputs

Clean only Aim runs and experiment metadata:

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
local/aim/
local/checkpoints/
local/figures/
```

## Format, lint, and test

Automatically fix lint issues where possible:

```bash
pixi run fix
```

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

A typical cleanup/check sequence before committing is:

```bash
pixi run fix
pixi run format
pixi run lint
pixi run pytest
```

## Extending the template

> [!TIP]
> Start by replacing the data module and model, then update `configs/default.yaml` to match your project.

Common next steps:

- Replace `LinearRegressionData` in `model/dataset.py` with your own data module.
- Replace `FCNet` in `model/model.py` with your own neural network model.
- Modify `mse_loss` in `model/loss.py` or add new loss functions.
- Add more configuration files under `configs/`.
- Add project-specific metrics, plots, callbacks, or Aim-tracked figures.
- Modify `AimFitPlotCallback` in `callbacks.py` for custom image logging.
- Add real unit tests under `tests/`.

## Notes

- `local/` is intentionally ignored by git.
- Keep raw data, generated data, Aim runs, checkpoints, and figures under `local/`.
- Commit `pixi.lock` for reproducible environments.
- The default example trains a tiny fully connected model on a synthetic linear regression dataset.
- The template uses local Aim tracking by default.
- Model checkpoints are saved locally rather than uploaded to a cloud registry.

## License

This project is licensed under the MIT License.

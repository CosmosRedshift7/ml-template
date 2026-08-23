# Pixi PyTorch Lightning ML Template

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10-ee4c2c)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.6-purple)](https://lightning.ai/docs/pytorch/latest/)
[![Aim](https://img.shields.io/badge/Aim-experiment%20tracking-111111)](https://aimstack.io/)
[![Pixi](https://img.shields.io/badge/Pixi-reproducible%20envs-f0b90b)](https://pixi.sh/)
[![CI](https://github.com/CosmosRedshift7/ml-template/actions/workflows/ci.yml/badge.svg)](https://github.com/CosmosRedshift7/ml-template/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> A minimal PyTorch Lightning template for reproducible AI, machine learning, and scientific computing that runs locally by default and can be extended to remote workflows—with locked Pixi environments, Aim experiment tracking, YAML configuration, and CPU/GPU training.

Reproducibility is especially important in scientific machine learning, including physics-informed and data-driven research. Even code published alongside manuscripts in strong journals can be difficult to reproduce because of dependency drift, undocumented commands, missing configurations, or unclear experiment histories. This template provides a compact structure for preserving the environment, configuration, execution commands, metrics, and outputs behind each experiment.

Experiments are tracked locally by default with Aim, without requiring Docker or cloud services, while the project can be adapted to shared or remote tracking workflows. Although designed with scientific projects in mind, the template is useful for any AI or ML project where repeatable execution, traceable experiments, and reproducible results matter.

[**Use this template →**](https://github.com/CosmosRedshift7/ml-template/generate) · [Documentation](https://cosmosredshift7.github.io/ml-template/)

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
| Supported hosts          | Linux, macOS, and Windows through WSL2 |
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

## Platform support

| Host                | CPU environment        | CUDA environment                                    |
| ------------------- | ---------------------- | --------------------------------------------------- |
| Linux x86-64        | Supported              | Supported with NVIDIA GPU                           |
| macOS Intel         | Supported              | Not available                                       |
| macOS Apple silicon | Supported              | Not available                                       |
| Windows 10/11       | Supported through WSL2 | Supported through WSL2 with a compatible NVIDIA GPU |

Aim supports Linux and macOS but not native Windows. Windows users should run the template inside [WSL2](https://learn.microsoft.com/windows/wsl/install), which uses the locked Linux environment. Native Windows is not currently supported by the complete template.

## Setup

Install Pixi first if you do not already have it.

Linux and macOS:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Windows: first install WSL2 from an administrator PowerShell terminal, then restart if prompted:

```powershell
wsl --install
```

Open the installed Ubuntu terminal, clone the repository there, and use the Linux Pixi installer shown above.

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
pixi clean
pixi install --frozen
```

To fully resolve dependencies again and regenerate the lock file:

```bash
pixi update
```

> [!WARNING] > `pixi clean` removes the local environments. `pixi update` can change locked package versions, so review and commit the resulting `pixi.lock` diff.

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
> GPU training requires Linux or WSL2, an NVIDIA GPU, a compatible NVIDIA driver, and the CUDA-enabled Pixi environment. CUDA is not available on macOS. The CPU environment is kept as the default because it works on most machines.

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

Use these checks before committing changes. `fix` applies automatic Ruff fixes where possible, `format` formats the code, `lint` checks for remaining style/import issues, and `pytest` runs the smoke tests.

```bash
pixi run fix
pixi run format
pixi run lint
pixi run pytest
```

### Starting a new project from this template

For a new research project, use the GitHub **Use this template** button. This creates a fresh repository with the same files but without carrying over the template commit history.

Alternatively, create a fresh local copy manually:

```bash
git clone https://github.com/CosmosRedshift7/ml-template.git new-project-name
cd new-project-name
rm -rf .git
git init
git add -A
git commit -m "Initial commit from ml-template"
```

After creating the new repository, update the project-specific files. At minimum, update the project metadata in `pyproject.toml`:

```toml
[project]
name = "new-project-name"
description = "Short description of the new project"
```

and update the Aim experiment name in `configs/default.yaml`:

```yaml
aim:
  experiment_name: new-project-name
```

> [!TIP]
> Fork this repository only if you want your new repository to remain visibly connected to `ml-template` or if you plan to contribute changes back to the template.

## Extending the template

> [!TIP]
> Start by replacing the data module and model, then update `configs/default.yaml` to match your project.

Common next steps:

- Replace `LinearRegressionData` in `model/dataset.py` with your own data module.
- Replace `FCNet` in `model/model.py` with your own neural network model.
- Modify `mse_loss` in `model/loss.py` or add new loss functions.
- Add more configuration files under `configs/`.
- Add project-specific metrics, plots, callbacks, or Aim-tracked figures.
- Modify `AimPlotCallback` in `callbacks.py` for custom image logging.
- Add real unit tests under `tests/`.

## Notes

- Keep raw data, generated data, Aim runs, checkpoints, and figures under `local/`.
- Commit `pixi.lock` for reproducible environments.
- The default example trains a tiny fully connected model on a synthetic linear regression dataset.
- The template uses local Aim tracking by default.

## License

This project is licensed under the MIT License.

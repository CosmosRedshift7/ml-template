# Pixi + PyTorch + Lightning + Aim ML Template

A lightweight, reproducible machine learning project template using **Pixi**, **PyTorch Lightning**, **Aim experiment tracking**, YAML configs, pytest smoke tests, and CPU/GPU-ready environments. The CPU environment is locked for Linux, Intel macOS, and Apple silicon macOS; Windows users run the Linux environment through WSL2.

This template is designed for research ML projects where you want a clean starting point with reproducible dependencies, structured training and evaluation scripts, local experiment tracking, and easy project reuse.

[Repository](https://github.com/CosmosRedshift7/ml-template)

## What this template gives you

| Area                | Included                           |
| ------------------- | ---------------------------------- |
| Environment         | Pixi + `pixi.lock`                 |
| Deep learning       | PyTorch                            |
| Training framework  | PyTorch Lightning                  |
| Experiment tracking | Local Aim tracking                 |
| Configuration       | YAML config files                  |
| Checkpointing       | Lightning `ModelCheckpoint`        |
| Evaluation          | Separate `evaluate.py` entry point |
| Plot logging        | Aim callback for figures           |
| Testing             | Pytest smoke tests                 |
| Code quality        | Ruff formatting and linting        |
| Local outputs       | Ignored `local/` directory         |

## Good fit

Use this template when you want:

- reproducible Python environments,
- clean training and evaluation entry points,
- config-driven experiments,
- local experiment tracking,
- simple checkpointing,
- smoke tests,
- and a project layout that is easy to copy into a new repository.

## Quick example

```bash
git clone https://github.com/CosmosRedshift7/ml-template.git
cd ml-template

pixi install
pixi run train
pixi run aim-ui
```

Then open:

```text
http://127.0.0.1:43800
```

## Keywords

Pixi PyTorch template, PyTorch Lightning template, machine learning template, ML research template, Aim experiment tracking, reproducible ML environment, Python deep learning template, GPU training template.

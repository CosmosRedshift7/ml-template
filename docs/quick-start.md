# Quick start

This page gives the shortest path from cloning the repository to seeing a tracked experiment in Aim.

The commands below run natively on Linux and macOS. On Windows, run them inside WSL2 because Aim does not support native Windows.

## Clone the repository

```bash
git clone https://github.com/CosmosRedshift7/ml-template.git
cd ml-template
```

## Install dependencies

```bash
pixi install
```

This creates a local Pixi environment from `pyproject.toml` and `pixi.lock`.

## Run training

```bash
pixi run train
```

This trains the example model and writes local outputs under:

```text
local/
```

Typical outputs include:

```text
local/aim/
local/checkpoints/
local/figures/
```

## Open Aim UI

```bash
pixi run aim-ui
```

Then open:

```text
http://127.0.0.1:43800
```

In Aim, open the `ml-template` experiment to view metrics, parameters, and tracked figures.

## Evaluate

After training, evaluate the best checkpoint:

```bash
pixi run evaluate
```

By default, this evaluates:

```text
local/checkpoints/best.ckpt
```

## Recommended first check

Before editing the template, run:

```bash
pixi run pytest
```

This confirms that the project imports correctly and the smoke tests pass.

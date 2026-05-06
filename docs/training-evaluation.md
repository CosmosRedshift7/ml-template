# Training and evaluation

The template separates training and evaluation into two entry points:

```text
train.py
evaluate.py
```

## Run training

```bash
pixi run train
```

Equivalent manual command:

```bash
pixi run python train.py --config configs/default.yaml
```

Training will:

- load `configs/default.yaml`,
- create the data module,
- create the model,
- train with PyTorch Lightning,
- track metrics and parameters with Aim,
- save checkpoints under `local/checkpoints/`,
- save figures under `local/figures/`.

## Training outputs

Generated outputs are written under:

```text
local/
```

This folder is ignored by git, so experiments, checkpoints, and figures do not pollute the repository history.

## Evaluate the default checkpoint

```bash
pixi run evaluate
```

By default, evaluation uses:

```text
local/checkpoints/best.ckpt
```

Run training first unless you already have that checkpoint.

## Evaluate a different checkpoint

```bash
pixi run python evaluate.py --config configs/default.yaml --ckpt path/to/checkpoint.ckpt
```

Evaluation logs test metrics and tracks a predicted-vs-true fit plot in Aim.

## Main files involved

| File | Purpose |
| --- | --- |
| `train.py` | Training entry point |
| `evaluate.py` | Evaluation entry point |
| `configs/default.yaml` | Experiment configuration |
| `model/dataset.py` | Data module |
| `model/model.py` | Neural network model |
| `model/pl_model.py` | Lightning module |
| `callbacks.py` | Aim figure logging callback |

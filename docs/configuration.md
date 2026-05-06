# Configuration

The main configuration file is:

```text
configs/default.yaml
```

It controls the dataset, model, optimizer, trainer, checkpoints, evaluation, and Aim tracking.

## Typical fields

The default config controls:

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

## Trainer settings

Example:

```yaml
trainer:
  max_epochs: 10
  accelerator: auto
  devices: auto
```

For CPU/GPU portability, `auto` is a good default.

For explicit single-GPU training:

```yaml
trainer:
  accelerator: gpu
  devices: 1
```

For distributed training:

```yaml
trainer:
  accelerator: gpu
  devices: 2
  strategy: ddp
```

## Aim settings

Example:

```yaml
aim:
  repo: local/aim
  experiment_name: ml-template
```

When starting a new project, change `experiment_name`.

## Checkpoint settings

The template saves checkpoints under:

```text
local/checkpoints/
```

A common default evaluation checkpoint is:

```text
local/checkpoints/best.ckpt
```

## Adding new configs

Add new YAML files under:

```text
configs/
```

For example:

```text
configs/debug.yaml
configs/gpu.yaml
configs/large_model.yaml
```

Then run:

```bash
pixi run python train.py --config configs/debug.yaml
```

# Experiment tracking with Aim

This template uses Aim for local experiment tracking.

## Start the Aim UI

```bash
pixi run aim-ui
```

Then open:

```text
http://127.0.0.1:43800
```

## Aim repository location

Aim data is stored locally under:

```text
local/aim/
```

This directory is ignored by git.

## Experiment name

The default experiment name is configured in:

```text
configs/default.yaml
```

Look for:

```yaml
aim:
  experiment_name: ml-template
```

When starting a new project from the template, change this to the new project name.

## What gets tracked

The template tracks:

- training loss,
- validation loss,
- test loss,
- selected hyperparameters,
- generated figures,
- and predicted-vs-true fit plots.

## Figure logging

Figure logging is handled by:

```text
callbacks.py
```

The default callback pattern is useful for logging diagnostic plots during training and evaluation.

## Cleaning Aim runs

To delete local Aim experiment data:

```bash
pixi run clean-runs
```

This removes:

```text
local/aim/
```

Only run this when you are sure you no longer need the local run history.

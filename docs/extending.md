# Extending the template

The fastest way to adapt the template is to replace the data module and model first, then update the config.

## Replace the data module

Edit:

```text
model/dataset.py
```

The default example uses a toy synthetic linear regression dataset. Replace it with your own PyTorch Lightning data module.

Common changes:

- load real data,
- define train/validation/test splits,
- set batch size from the config,
- return PyTorch `DataLoader` objects.

## Replace the model

Edit:

```text
model/model.py
```

Replace the default fully connected network with your project model.

Examples:

- MLP,
- CNN,
- Transformer,
- Fourier Neural Operator,
- custom scientific ML architecture.

## Update the Lightning module

Edit:

```text
model/pl_model.py
```

Use this file to control:

- training step,
- validation step,
- test step,
- optimizer setup,
- metric logging.

## Add or change loss functions

Edit:

```text
model/loss.py
```

You can add project-specific losses here and call them from the Lightning module.

## Add custom plots

Edit:

```text
callbacks.py
```

Use the existing Aim callback as a pattern for logging figures during training or evaluation.

## Add new configs

Create files under:

```text
configs/
```

For example:

```text
configs/debug.yaml
configs/gpu.yaml
configs/production.yaml
```

Then run:

```bash
pixi run python train.py --config configs/debug.yaml
```

## Add tests

Add real unit tests under:

```text
tests/
```

Keep at least one smoke test that checks basic imports and minimal execution.

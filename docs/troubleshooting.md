# Troubleshooting

Common problems and fixes.

## `pixi: command not found`

Restart your terminal after installing Pixi.

If it still fails, check that Pixi's binary directory is in your shell `PATH`.

## Environment seems broken

Rebuild from the lock file:

```bash
pixi clean
pixi install --frozen
```

If you intentionally want to re-resolve dependencies:

```bash
pixi update
```

Be careful: `pixi update` can produce different dependency versions. Review the `pixi.lock` diff before committing it.

## Native Windows install cannot resolve Aim

Aim does not support native Windows. Install WSL2 with `wsl --install`, open the installed Ubuntu terminal, and run the project there.

## CUDA is not available

Check CUDA from the GPU environment:

```bash
pixi run -e gpu python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'
```

If this prints `False`, check:

- you are using the GPU environment,
- your machine has an NVIDIA GPU,
- the NVIDIA driver is installed,
- the driver is compatible with the CUDA-enabled PyTorch package.

## `pixi run evaluate` cannot find checkpoint

Run training first:

```bash
pixi run train
```

Then evaluate:

```bash
pixi run evaluate
```

Or pass a checkpoint manually:

```bash
pixi run python evaluate.py --config configs/default.yaml --ckpt path/to/checkpoint.ckpt
```

## Aim UI opens but no runs appear

Make sure you trained first:

```bash
pixi run train
```

Then start Aim:

```bash
pixi run aim-ui
```

Open:

```text
http://127.0.0.1:43800
```

Look for the `ml-template` experiment.

## Local files are not committed

This is expected for generated outputs under:

```text
local/
```

The `local/` directory is ignored by git to keep experiment outputs, checkpoints, figures, and temporary files out of version control.

## Tests fail after modifying the template

Run:

```bash
pixi run pytest
```

Then check whether the failure is from:

- renamed modules,
- changed config keys,
- missing imports,
- removed example files,
- or changed expected output paths.

Update the smoke tests after major structural changes.

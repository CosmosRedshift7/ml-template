# GPU training

The template supports both CPU and CUDA-enabled GPU environments through Pixi.

## Environments

```text
cpu      # CPU PyTorch environment
gpu      # CUDA-enabled PyTorch environment
default  # CPU environment by default
```

The CPU environment is the default because it works on most machines.

## Run with CPU

```bash
pixi run train
```

or explicitly:

```bash
pixi run -e cpu train
```

## Install the GPU environment

```bash
pixi install -e gpu
```

## Check CUDA availability

```bash
pixi run -e gpu python -c 'import torch; print(torch.cuda.is_available()); print(torch.version.cuda)'
```

Expected output should look similar to:

```text
True
12.9
```

The exact CUDA version may depend on the package versions and your environment.

## Run training with GPU

```bash
pixi run -e gpu train
```

With the default Lightning settings,

```yaml
trainer:
  accelerator: auto
  devices: auto
```

Lightning will use a GPU automatically when one is available.

## Explicit single-GPU config

In `configs/default.yaml`:

```yaml
trainer:
  accelerator: gpu
  devices: 1
```

Then run:

```bash
pixi run -e gpu train
```

## Multi-GPU config

For two GPUs with distributed data parallel training:

```yaml
trainer:
  accelerator: gpu
  devices: 2
  strategy: ddp
```

For all available GPUs:

```yaml
trainer:
  accelerator: gpu
  devices: auto
  strategy: ddp
```

## Requirements

GPU training requires:

- an NVIDIA GPU,
- a compatible NVIDIA driver,
- the CUDA-enabled Pixi environment,
- and PyTorch with CUDA support.

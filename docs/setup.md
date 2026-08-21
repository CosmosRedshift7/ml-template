# Installation and Pixi setup

This project uses Pixi for reproducible environments and one-command task execution.

## Install Pixi

Linux and macOS:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Windows: install WSL2 from an administrator PowerShell terminal:

```powershell
wsl --install
```

Restart if prompted, open the installed Ubuntu terminal, and use the Linux Pixi installer shown above. Aim supports Linux and macOS but not native Windows, so the complete template is supported on Windows through WSL2.

!!! important

    Restart your terminal or shell after installing Pixi so that the `pixi` command becomes available.

## Install the project environment

From the repository root:

```bash
pixi install
```

Pixi reads project dependencies from `pyproject.toml` and resolves them using `pixi.lock`.

## Why commit `pixi.lock`

Commit `pixi.lock` so that another machine can recreate the same package versions. This reduces dependency drift between machines.

## Activate the environment

For interactive work:

```bash
pixi shell
```

Inside the shell, commands such as `python`, `pytest`, and `ruff` run inside the project environment.

## Rebuild the local environment

To rebuild from the existing lock file:

```bash
pixi clean
pixi install --frozen
```

To force a full dependency re-resolution:

```bash
pixi update
```

!!! warning

    `pixi clean` removes the local environments. `pixi update` can change locked package versions, so review and commit the resulting `pixi.lock` diff.

## Supported platforms

| Host | CPU environment | CUDA environment |
| --- | --- | --- |
| Linux x86-64 | Supported | Supported with NVIDIA GPU |
| macOS Intel | Supported | Not available |
| macOS Apple silicon | Supported | Not available |
| Windows 10/11 | Supported through WSL2 | Supported through WSL2 with a compatible NVIDIA GPU |

# Installation and Pixi setup

This project uses Pixi for reproducible environments and one-command task execution.

## Install Pixi

Linux and macOS:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Windows installer:

```text
https://github.com/prefix-dev/pixi/releases/latest/download/pixi-x86_64-pc-windows-msvc.msi
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

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
rm -rf .pixi
pixi install
```

To force a full dependency re-resolution:

```bash
rm -rf .pixi pixi.lock
pixi install
```

!!! warning

    Deleting `.pixi/` removes the local environment. Deleting `pixi.lock` allows dependency versions to change.

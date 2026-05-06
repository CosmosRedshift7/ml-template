# Development workflow

This page summarizes the common development commands.

## Format code

```bash
pixi run format
```

## Apply automatic Ruff fixes

```bash
pixi run fix
```

## Lint code

```bash
pixi run lint
```

## Run tests

```bash
pixi run pytest
```

## Recommended pre-commit check

Before committing:

```bash
pixi run fix
pixi run format
pixi run lint
pixi run pytest
```

## Clean local outputs

Clean Aim runs:

```bash
pixi run clean-runs
```

Clean checkpoints:

```bash
pixi run clean-checkpoints
```

Clean generated figures:

```bash
pixi run clean-figures
```

Clean all generated local outputs:

```bash
pixi run clean-all
```

These commands remove files under:

```text
local/aim/
local/checkpoints/
local/figures/
```

## Commit changes

```bash
git status
git add -A
git commit -m "Describe the change"
git push
```

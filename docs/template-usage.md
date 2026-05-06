# Using this as a template

This repository is designed to be copied into new ML research projects.

## Recommended method: GitHub template

On GitHub, click:

```text
Use this template
```

This creates a fresh repository with the same files but without carrying over the original commit history.

## Manual local method

```bash
git clone https://github.com/CosmosRedshift7/ml-template.git new-project-name
cd new-project-name
rm -rf .git
git init
git add -A
git commit -m "Initial commit from ml-template"
```

Then create a new GitHub repository and push to it.

## Files to update in a new project

At minimum, update `pyproject.toml`:

```toml
[project]
name = "new-project-name"
description = "Short description of the new project"
```

Update the Aim experiment name in `configs/default.yaml`:

```yaml
aim:
  experiment_name: new-project-name
```

Update the README title and description:

```md
# New Project Name
```

## Fork vs template

Use **Use this template** when creating a new independent project.

Use **Fork** only if you want the new repository to remain visibly connected to `ml-template` or if you plan to contribute changes back to the template.

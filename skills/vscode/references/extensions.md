# Recommended Extensions

Scope: the `.vscode/extensions.json` recommendation list for a CV/ML project and what
each extension is for.

## .vscode/extensions.json

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-python.mypy-type-checker",
        "ms-toolsai.jupyter",
        "ms-toolsai.jupyter-renderers",
        "ms-azuretools.vscode-docker",
        "ms-vscode-remote.remote-ssh",
        "ms-vscode-remote.remote-containers",
        "eamodio.gitlens",
        "tamasfe.even-better-toml",
        "redhat.vscode-yaml",
        "GitHub.copilot",
        "ms-python.debugpy"
    ]
}
```

Committing this file makes VS Code prompt teammates to install the full toolchain the
first time they open the workspace.

## Extension Descriptions

| Extension | Purpose |
|-----------|---------|
| `ms-python.python` | Core Python support |
| `ms-python.vscode-pylance` | Fast type checking and IntelliSense |
| `charliermarsh.ruff` | Ruff linting and formatting |
| `ms-python.mypy-type-checker` | Mypy integration |
| `ms-toolsai.jupyter` | Notebook support |
| `ms-azuretools.vscode-docker` | Dockerfile editing and container management |
| `ms-vscode-remote.remote-ssh` | Remote development on GPU servers |
| `ms-vscode-remote.remote-containers` | Dev Containers |
| `eamodio.gitlens` | Git history and blame |
| `tamasfe.even-better-toml` | TOML syntax for pixi.toml and pyproject.toml |

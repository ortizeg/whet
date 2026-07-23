---
name: vscode
description: >
  Use this skill when configuring VS Code for CV/ML development — workspace settings.json,
  Ruff and MyPy editor integration, Python debug launch configs, tasks, remote-SSH to GPU
  servers, dev containers, and recommended extensions. Reach for it any time you'd
  otherwise hand-edit .vscode files or set up a remote GPU dev environment, even if the
  user just says "set up my editor" or "debug this in VS Code". This is editor setup; the
  lint/type standards live in code-quality and commit hooks in pre-commit.
---

# VS Code Skill

Configure VS Code for productive computer vision and machine learning development with integrated linting, debugging, and remote GPU server support.

## Workspace Settings

### .vscode/settings.json

```jsonc
{
    // Python
    "python.defaultInterpreterPath": "${workspaceFolder}/.pixi/envs/default/bin/python",
    "python.analysis.typeCheckingMode": "strict",
    "python.analysis.autoImportCompletions": true,

    // Ruff (replaces black, isort, flake8)
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "ruff.lineLength": 100,

    // mypy
    "mypy-type-checker.args": [
        "--strict",
        "--config-file=${workspaceFolder}/pyproject.toml"
    ],
    "python.analysis.diagnosticMode": "workspace",

    // Testing
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests",
        "-v",
        "--tb=short"
    ],
    "python.testing.unittestEnabled": false,

    // File associations
    "files.associations": {
        "*.yaml": "yaml",
        "*.yml": "yaml",
        "Dockerfile*": "dockerfile",
        "*.toml": "toml"
    },

    // Exclude large directories from explorer and search
    "files.exclude": {
        "**/__pycache__": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true,
        "**/.pytest_cache": true,
        "**/htmlcov": true,
        "**/site": true,
        "**/*.egg-info": true
    },
    "search.exclude": {
        "**/data": true,
        "**/checkpoints": true,
        "**/outputs": true,
        "**/wandb": true,
        "**/mlruns": true,
        "**/lightning_logs": true,
        "**/.pixi": true
    },

    // Editor
    "editor.rulers": [100],
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,

    // Terminal
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.env.linux": {
        "PATH": "${workspaceFolder}/.pixi/envs/default/bin:${env:PATH}"
    },

    // Jupyter
    "jupyter.notebookFileRoot": "${workspaceFolder}",
    "notebook.formatOnSave.enabled": true,
    "notebook.codeActionsOnSave": {
        "source.fixAll.ruff": "explicit"
    }
}
```

## Recommended Extensions

### .vscode/extensions.json

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

### Extension Descriptions

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

## Debug Configurations

### .vscode/launch.json

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        // =====================================================================
        // Training — debug a Lightning training run
        // =====================================================================
        {
            "name": "Train: Debug",
            "type": "debugpy",
            "request": "launch",
            "module": "my_project.train",
            "args": [
                "trainer.max_epochs=2",
                "trainer.fast_dev_run=true",
                "data.batch_size=4"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },

        // =====================================================================
        // Tests — debug pytest with current file or specific test
        // =====================================================================
        {
            "name": "Test: Current File",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": [
                "${file}",
                "-v",
                "--tb=short",
                "--no-header"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },
        {
            "name": "Test: All",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/",
                "-v",
                "--tb=short"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },

        // =====================================================================
        // Inference — debug FastAPI serving endpoint
        // =====================================================================
        {
            "name": "Serve: FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "my_project.serve:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ],
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src",
                "MODEL_PATH": "${workspaceFolder}/models/best.onnx"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        },

        // =====================================================================
        // Script — debug any Python script
        // =====================================================================
        {
            "name": "Script: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            },
            "justMyCode": false,
            "console": "integratedTerminal"
        }
    ]
}
```

## Task Definitions

### .vscode/tasks.json

```jsonc
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Lint",
            "type": "shell",
            "command": "pixi run lint",
            "group": "test",
            "problemMatcher": ["$eslint-stylish"],
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Format",
            "type": "shell",
            "command": "pixi run format",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Type Check",
            "type": "shell",
            "command": "pixi run typecheck",
            "group": "test",
            "problemMatcher": ["$tsc"],
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Test",
            "type": "shell",
            "command": "pixi run test",
            "group": {
                "kind": "test",
                "isDefault": true
            },
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        },
        {
            "label": "Train",
            "type": "shell",
            "command": "pixi run python -m my_project.train",
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always"
            }
        }
    ]
}
```

## Python Interpreter with Pixi

Configure VS Code to use the pixi-managed Python:

```jsonc
{
    // Auto-detect pixi environment
    "python.defaultInterpreterPath": "${workspaceFolder}/.pixi/envs/default/bin/python",

    // Alternative: use pixi run prefix for terminal commands
    "terminal.integrated.env.linux": {
        "PATH": "${workspaceFolder}/.pixi/envs/default/bin:${env:PATH}"
    },
    "terminal.integrated.env.osx": {
        "PATH": "${workspaceFolder}/.pixi/envs/default/bin:${env:PATH}"
    }
}
```

## Remote Development

### SSH to GPU Server

```jsonc
// .vscode/settings.json (for remote SSH)
{
    "remote.SSH.defaultExtensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-python.mypy-type-checker",
        "ms-toolsai.jupyter"
    ],
    "remote.SSH.configFile": "~/.ssh/config"
}
```

SSH config example:

```
# ~/.ssh/config
Host gpu-server
    HostName 192.168.1.100
    User researcher
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    LocalForward 6006 localhost:6006    # TensorBoard
    LocalForward 8000 localhost:8000    # Inference API
    LocalForward 5000 localhost:5000    # MLflow UI
```

### Dev Containers

```jsonc
// .devcontainer/devcontainer.json
{
    "name": "ML Dev Container",
    "build": {
        "dockerfile": "../Dockerfile",
        "target": "training"
    },
    "runArgs": [
        "--gpus", "all",
        "--shm-size", "8g"
    ],
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "charliermarsh.ruff",
                "ms-toolsai.jupyter"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/app/.pixi/envs/default/bin/python"
            }
        }
    },
    "forwardPorts": [6006, 8000],
    "postCreateCommand": "pixi install"
}
```

## Workspace vs User Settings

| Setting | Scope | Location |
|---------|-------|----------|
| Python interpreter path | Workspace | `.vscode/settings.json` |
| Ruff configuration | Workspace | `.vscode/settings.json` |
| File exclusions | Workspace | `.vscode/settings.json` |
| Theme, font size | User | `~/.config/Code/User/settings.json` |
| Keybindings | User | `~/.config/Code/User/keybindings.json` |
| Extension sync | User | VS Code settings sync |

**Rule:** Workspace settings go in `.vscode/` and are committed to git. User preferences stay in your user settings and are never committed.

## Best Practices

1. **Commit `.vscode/`** -- share workspace settings, launch configs, and extension recommendations
2. **Don't commit user settings** -- themes, fonts, and personal keybindings stay user-level
3. **Use pixi interpreter** -- ensures everyone uses the same Python and packages
4. **Configure search exclusions** -- hide data/, checkpoints/, and wandb/ from search
5. **Set up debug configs** -- pre-configure training, testing, and serving debug sessions
6. **Use tasks** -- define pixi run commands as VS Code tasks for quick access
7. **Install recommended extensions** -- prompt teammates on first open
8. **Configure remote SSH** -- port-forward TensorBoard and MLflow for remote GPU development
9. **Use Dev Containers** -- consistent GPU environment across team members
10. **Set rulers at line length** -- visual guide matching Ruff's `line-length = 100`

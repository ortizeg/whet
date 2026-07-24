# Dev Containers

Scope: a GPU-enabled `.devcontainer/devcontainer.json` that reuses the project's
multi-stage Dockerfile so every team member gets an identical CUDA environment.

## .devcontainer/devcontainer.json

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

## Notes

- `"target": "training"` selects the training stage of the project's multi-stage
  Dockerfile, so the dev container matches what CI and production build from.
- `--gpus all` is required for CUDA visibility inside the container; without it
  `torch.cuda.is_available()` returns `False`.
- `--shm-size 8g` raises shared memory above Docker's 64 MB default — PyTorch DataLoader
  workers crash with "bus error" or "DataLoader worker killed" otherwise.
- `forwardPorts` exposes TensorBoard (6006) and the inference API (8000) to the host.
- `postCreateCommand: "pixi install"` materializes the environment on first container
  create rather than baking it into every image layer.

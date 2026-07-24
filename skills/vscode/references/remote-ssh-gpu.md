# Remote SSH to GPU Servers

Scope: configuring VS Code Remote-SSH for development on a GPU box, including port
forwarding for TensorBoard, MLflow, and an inference API.

## VS Code Remote-SSH Settings

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

`remote.SSH.defaultExtensions` installs the listed extensions automatically on every new
remote host, so a fresh GPU server is usable immediately after connecting.

## SSH Config

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

`LocalForward` entries make remote dashboards reachable at `localhost` on your laptop —
open `http://localhost:6006` for TensorBoard running on the GPU box. `ForwardAgent yes`
lets the remote host use your local SSH keys for git operations without copying keys onto
the server.

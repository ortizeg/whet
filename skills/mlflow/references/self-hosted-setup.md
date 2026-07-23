# Self-Hosted Setup and Tracking URIs

Scope: installing MLflow, running the local UI, standing up a remote tracking server, and choosing between local and remote tracking.

## Contents

- [Installation](#installation)
- [Local tracking server](#local-tracking-server)
- [Remote tracking server](#remote-tracking-server)
- [Local vs remote tracking](#local-vs-remote-tracking)

## Installation

```bash
# Using pip
pip install mlflow

# Using pixi
pixi add mlflow --feature experiment-tracking

# With extras for specific backends
pip install mlflow[extras]
```

## Local tracking server

For local development, MLflow stores data in the `mlruns` directory by default:

```bash
# Start the MLflow UI (local tracking)
mlflow ui --port 5000

# Access at http://localhost:5000
```

## Remote tracking server

For team collaboration, set up a remote tracking server:

```bash
# Start server with PostgreSQL backend and S3 artifact store
mlflow server \
    --backend-store-uri postgresql://user:pass@localhost:5432/mlflow \
    --default-artifact-root s3://my-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

Configure the client to point to the remote server:

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server:5000")
```

Or use an environment variable:

```bash
export MLFLOW_TRACKING_URI=http://mlflow-server:5000
```

## Local vs remote tracking

### Local (default)

```python
# Data stored in ./mlruns directory
import mlflow
mlflow.set_tracking_uri("file:///path/to/mlruns")
```

Best for: Individual development, quick experiments, offline work.

### Remote

```python
import mlflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

Best for: Team collaboration, centralized experiment management, production workflows.

Prefer setting `MLFLOW_TRACKING_URI` in the environment over hardcoding
`set_tracking_uri` in application code, so the same script runs locally and against
the team server without edits. Add `.mlruns/` to `.gitignore` so local tracking data
never lands in version control.

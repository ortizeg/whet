# Artifact Registry

Scope: creating and using Artifact Registry repositories for Docker images and Python packages.

Use Artifact Registry (replaces the deprecated Container Registry) for Docker images and Python packages.

## Create a Docker Repository

```bash
# Create a Docker repository in Artifact Registry
gcloud artifacts repositories create ml-images \
    --repository-format=docker \
    --location=us-central1 \
    --description="CV/ML training and inference images"

# Verify creation
gcloud artifacts repositories list --location=us-central1
```

## Push and Pull Docker Images

```bash
# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

# Tag and push
docker tag my-training:latest \
    us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0

docker push us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0

# Pull
docker pull us-central1-docker.pkg.dev/my-project/ml-images/training:v1.2.0
```

## Python Package Repository

```bash
gcloud artifacts repositories create ml-packages \
    --repository-format=python --location=us-central1

# Print pip/uv index settings for the repo
gcloud artifacts print-settings python --repository=ml-packages --location=us-central1
```

Add the printed index to `pyproject.toml` under `[tool.uv]` as an `extra-index-url` (e.g. `https://us-central1-python.pkg.dev/my-project/ml-packages/simple/`).

# Cloud Storage

Scope: moving datasets, checkpoints, and training artifacts in and out of GCS with gsutil, the Python client, and gcsfuse.

Use Cloud Storage for datasets, model checkpoints, and training artifacts.

## Upload and Download with gsutil

```bash
# Upload a dataset
gsutil -m cp -r ./data/coco/ gs://my-ml-bucket/datasets/coco/

# Download model checkpoint
gsutil cp gs://my-ml-bucket/checkpoints/resnet50_epoch20.pt ./checkpoints/

# Sync outputs (only transfers changed files)
gsutil -m rsync -r ./outputs/ gs://my-ml-bucket/runs/experiment-42/
```

## Python Client

```python
from google.cloud import storage


def upload_model_artifact(
    bucket_name: str,
    source_path: str,
    destination_blob: str,
) -> str:
    """Upload a model artifact to Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_path)
    return f"gs://{bucket_name}/{destination_blob}"


def download_checkpoint(
    bucket_name: str,
    blob_name: str,
    destination_path: str,
) -> None:
    """Download a checkpoint from Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_path)
```

## Mount with gcsfuse

```bash
# Mount a bucket as a local directory (useful for large datasets)
gcsfuse --implicit-dirs my-ml-bucket /mnt/gcs-data

# Use in a training container
docker run --privileged \
    -v /mnt/gcs-data:/data:ro \
    my-training:latest
```

```dockerfile
# Install gcsfuse in a training container
RUN echo "deb https://packages.cloud.google.com/apt gcsfuse-jammy main" \
        > /etc/apt/sources.list.d/gcsfuse.list && \
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key add - && \
    apt-get update && \
    apt-get install -y gcsfuse && \
    rm -rf /var/lib/apt/lists/*
```

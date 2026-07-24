# Docker Compose for CV/ML Stacks

Scope: composing multi-service local stacks — GPU training, TensorBoard, and an inference
service — with the volume, shared-memory, and device reservations CV workloads need.

## Training, Monitoring, and Inference

```yaml
# docker-compose.yml
services:
  train:
    build:
      context: .
      dockerfile: Dockerfile
      target: training
    volumes:
      - ./data:/app/data:ro          # Read-only data mount
      - ./checkpoints:/app/checkpoints  # Writable checkpoint output
      - ./configs:/app/configs:ro    # Config overrides
    environment:
      - WANDB_API_KEY=${WANDB_API_KEY}
      - CUDA_VISIBLE_DEVICES=0,1
    shm_size: "8gb"  # Required for DataLoader num_workers > 0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: ["my_project.train", "experiment=baseline"]

  tensorboard:
    image: tensorflow/tensorflow:latest
    ports:
      - "6006:6006"
    volumes:
      - ./outputs/logs:/logs:ro
    command: ["tensorboard", "--logdir=/logs", "--bind_all"]

  inference:
    build:
      context: .
      dockerfile: Dockerfile
      target: inference
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Notes on the Pieces

- **`target:`** selects the multi-stage build stage, so training and inference share one
  Dockerfile and one cached dependency layer.
- **Volumes** — data and configs are mounted `:ro`; only checkpoint output is writable.
  Datasets and checkpoints are never baked into the image.
- **`shm_size: "8gb"`** — the Compose default of 64 MB is too small for PyTorch
  `DataLoader` workers.
- **`deploy.resources.reservations.devices`** is the Compose v2 form of `--gpus`; use
  `count: all` for training and `count: 1` for a single inference replica.
- **Secrets** come from the environment (`${WANDB_API_KEY}`) or an `env_file`, never from
  a value committed into the Compose file.

## Common Commands

```bash
# Build and run the training service
docker compose up train

# Rebuild after dependency changes
docker compose build --no-cache train

# Run the inference stack in the background
docker compose up -d inference

# Tail logs
docker compose logs -f train
```

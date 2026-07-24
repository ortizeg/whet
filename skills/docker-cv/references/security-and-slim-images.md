# Container Security and Slim Images

Scope: hardening CV containers — non-root users, keeping secrets out of image layers,
read-only filesystems, and health checks for training and inference containers.

## Non-Root User

```dockerfile
# Always create and switch to a non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
```

Create the user after installing system packages (which need root) and before the
`ENTRYPOINT`. UID 1000 matches the typical host user, so bind-mounted volumes stay
writable without `chown` gymnastics.

## No Secrets in Images

```dockerfile
# ❌ WRONG: Secret baked into image layer
ENV WANDB_API_KEY=my-secret-key
COPY .env /app/.env

# ✅ CORRECT: Pass at runtime
# docker run -e WANDB_API_KEY=$WANDB_API_KEY my-image
# docker-compose with env_file or environment
```

Deleting a secret in a later layer does not remove it — every layer is retained in the
image and readable with `docker history`. For build-time credentials (private package
indexes) use BuildKit secret mounts:

```dockerfile
RUN --mount=type=secret,id=pip_token \
    pip install --extra-index-url "https://$(cat /run/secrets/pip_token)@pypi.internal/simple" mypkg
```

## Read-Only Filesystem

```yaml
# docker-compose.yml
services:
  inference:
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - ./models:/app/models:ro
```

Inference containers should never need to write to their own filesystem. Mount model
weights read-only and give scratch space through `tmpfs`.

## Health Checks

```dockerfile
# HTTP health check for API services
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# File-based health check for training containers
HEALTHCHECK --interval=60s --timeout=5s --retries=3 \
    CMD test -f /tmp/training_alive || exit 1
```

`--start-period` matters for model servers: weight loading can take a minute, and without
it the container is marked unhealthy while still warming up.

## Slim Inference Images

Keep the serving image minimal:

- Start from `python:3.11-slim` (or a CUDA `runtime` image when GPU inference is required)
  rather than a `devel` image — the CUDA SDK is build-time only.
- Install from a dedicated `requirements-inference.txt` instead of the dev dependency set.
- Use `pip install --no-cache-dir` so the wheel cache is not baked into the layer.
- Install only the system libraries OpenCV actually needs at runtime
  (`libgl1-mesa-glx`, `libglib2.0-0`), not the full training set.
- Copy model weights explicitly rather than the whole repository.

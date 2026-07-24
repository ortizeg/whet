# Docker Build and Push

Scope: building a multi-stage CV/ML image in CI and pushing it to GitHub Container
Registry with GitHub Actions layer caching.

## Build and Push to GHCR

Build a multi-stage image (`target: inference`) and push to GHCR with GHA layer caching:

```yaml
# .github/workflows/docker.yml
name: Build Docker Image
on:
  push:
    branches: [main]
    tags: ["v*"]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - name: Login to GHCR
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: |
            ghcr.io/${{ github.repository }}:${{ github.sha }}
            ghcr.io/${{ github.repository }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
          target: inference
```

## Notes

- `cache-from`/`cache-to: type=gha` reuses layers across runs; `mode=max` also caches
  intermediate build stages, which matters for heavy CUDA base layers.
- Tag with both `${{ github.sha }}` (immutable, deployable) and `latest` (convenience).
- `target: inference` selects the slim runtime stage of a multi-stage Dockerfile so the
  pushed image does not carry build tooling.

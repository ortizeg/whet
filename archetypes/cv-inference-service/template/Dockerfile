# syntax=docker/dockerfile:1
FROM python:${python_version}-slim

# pixi is bootstrapped over HTTPS, so a slim base needs curl + CA certificates.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://pixi.sh/install.sh | bash
ENV PATH="/root/.pixi/bin:$${PATH}"

WORKDIR /app

# Only pixi.toml ships with the template. Run `pixi install` locally, commit the
# generated pixi.lock, then add it here (`COPY pixi.toml pixi.lock ./`) and switch
# to `pixi install --locked` for fully reproducible builds.
COPY pixi.toml ./
RUN pixi install

COPY pyproject.toml README.md ./
COPY src/ ./src/

ENV PYTHONPATH=/app/src \
    PIXI_CACHE_DIR=/tmp/pixi-cache \
    MODEL_PATH=/app/models/model.onnx

# Run as an unprivileged user; the pixi bootstrap lives under /root, so make it
# traversable and readable (never writable) for that user.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/models /tmp/pixi-cache \
    && chown -R appuser:appuser /app /tmp/pixi-cache \
    && chmod a+rx /root \
    && chmod -R a+rX /root/.pixi
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["pixi", "run", "serve"]

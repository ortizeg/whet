# Pixi Dependency Management

Scope: choosing between conda-forge and PyPI sources for a package, declaring CUDA and
other system libraries, and version-pinning strategy.

## Conda vs PyPI Dependencies

Pixi supports both conda-forge and PyPI packages. Use conda-forge for system-level
packages and scientific libraries. Use PyPI for Python-only packages that are not on
conda-forge or where the PyPI version is newer.

```toml
# Conda dependencies (from conda-forge or pytorch channels)
[dependencies]
python = ">=3.11"
numpy = ">=1.26"
opencv = ">=4.9"              # System library, better from conda
ffmpeg = ">=6.0"              # System library, must be from conda
cuda-toolkit = ">=12.1"       # NVIDIA toolkit, only from conda

# PyPI dependencies (Python-only packages)
[pypi-dependencies]
torch = ">=2.2"               # PyTorch recommends pip install
lightning = ">=2.2"
albumentations = ">=1.3"
wandb = ">=0.16"
pydantic = ">=2.6"
timm = ">=0.9"
```

## Rules for Choosing Conda vs PyPI

1. **System libraries** (OpenCV, FFmpeg, CUDA): Always use conda.
2. **PyTorch ecosystem** (torch, torchvision, torchaudio): Use PyPI for reliable CUDA support.
3. **Pure Python packages** (pydantic, wandb, lightning): Either works; prefer PyPI for latest versions.
4. **Packages with C extensions** (Pillow, numpy, scipy): conda-forge often provides better-optimized builds.

## CUDA Toolkit Setup

CUDA is only available through conda, so it belongs in a `[feature.cuda.dependencies]`
block that machines without a GPU can skip:

```toml
[feature.cuda.dependencies]
cuda-toolkit = ">=12.1"

[feature.cuda.pypi-dependencies]
torch = { version = ">=2.2", extras = ["cuda"] }
```

Then compose a GPU environment from that feature so CI can install a CPU-only environment
while training machines install the CUDA one:

```toml
[environments]
default = { features = ["dev"], solve-group = "default" }
train = { features = ["dev", "cuda"], solve-group = "default" }
```

## Version Pinning Strategy

```toml
# Pin major version for stability (allow minor/patch updates)
numpy = ">=1.26,<2"

# Pin minimum version only (for most packages)
pydantic = ">=2.6"

# Exact pin only for reproducibility-critical packages
# (use sparingly -- lock files handle this)
python = "3.11.*"
```

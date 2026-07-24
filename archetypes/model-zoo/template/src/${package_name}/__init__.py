"""${project_name} — a curated zoo of pretrained models.

Public API:

    from ${package_name} import load_registry, download_weights

    registry = load_registry("registry")
    card = registry.get("resnet50")
    weights = download_weights(card)          # SHA-256 verified
"""

from ${package_name}.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    run_benchmark,
)
from ${package_name}.download import (
    ChecksumMismatchError,
    cached_path,
    download_weights,
    sha256_file,
    verify_file,
)
from ${package_name}.model_card import (
    Evaluation,
    InputSpec,
    ModelCard,
    WeightsSpec,
)
from ${package_name}.registry import (
    DuplicateModelError,
    ModelNotFoundError,
    ModelRegistry,
    RegistryError,
    load_registry,
)

__version__ = "0.1.0"

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "ChecksumMismatchError",
    "DuplicateModelError",
    "Evaluation",
    "InputSpec",
    "ModelCard",
    "ModelNotFoundError",
    "ModelRegistry",
    "RegistryError",
    "WeightsSpec",
    "__version__",
    "cached_path",
    "download_weights",
    "load_registry",
    "run_benchmark",
    "sha256_file",
    "verify_file",
]

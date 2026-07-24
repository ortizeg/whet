"""Shared fixtures.

Everything here is offline and pure-Python: no network, no torch, no GPU.
Weight "downloads" are exercised through ``file://`` URLs pointing at files the
fixtures create, which is enough to prove the SHA-256 verification path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from ${package_name}.model_card import ModelCard
from ${package_name}.registry import ModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def registry_dir() -> Path:
    """Path to the version-controlled model-card registry."""
    return PROJECT_ROOT / "registry"


@pytest.fixture(scope="session")
def card_paths(registry_dir: Path) -> list[Path]:
    """Every YAML model card shipped in the registry."""
    paths = sorted(p for p in registry_dir.rglob("*.yaml") if p.is_file())
    assert paths, f"no model cards found under {registry_dir}"
    return paths


@pytest.fixture
def registry(registry_dir: Path) -> ModelRegistry:
    """The shipped registry, fully loaded and validated."""
    return ModelRegistry.from_directory(registry_dir)


@pytest.fixture
def card_template() -> dict[str, object]:
    """A minimal, valid model-card payload for mutation in tests."""
    return {
        "name": "tiny-net",
        "version": "0.1.0",
        "task": "classification",
        "architecture": "tinynet",
        "license": "MIT",
        "training_dataset": "CIFAR-10",
        "inputs": {"channels": 3, "height": 32, "width": 32},
        "weights": {
            "url": "https://models.example.com/tiny-net.pth",
            "sha256": "0" * 64,
        },
    }


@pytest.fixture
def local_weights(tmp_path: Path) -> Iterator[Path]:
    """A real file on disk standing in for a hosted weight artefact."""
    path = tmp_path / "weights" / "tiny-net.pth"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"pretend-weight-bytes" * 512)
    yield path


@pytest.fixture
def make_local_card(
    local_weights: Path, card_template: dict[str, object]
) -> Callable[..., ModelCard]:
    """Build a card whose weights URL is a ``file://`` path to ``local_weights``."""

    def factory(*, sha256: str | None = None) -> ModelCard:
        digest = sha256 or hashlib.sha256(local_weights.read_bytes()).hexdigest()
        payload = dict(card_template)
        payload["weights"] = {"url": local_weights.as_uri(), "sha256": digest}
        return ModelCard.model_validate(payload)

    return factory

"""Registry loading, lookup, and filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ${package_name}.model_card import ModelCard
from ${package_name}.registry import (
    DuplicateModelError,
    ModelNotFoundError,
    ModelRegistry,
    RegistryError,
)


def test_registry_loads_shipped_cards(registry: ModelRegistry) -> None:
    assert len(registry) >= 2
    assert "resnet50" in registry
    assert "yolov8n" in registry


def test_lookup_by_name(registry: ModelRegistry) -> None:
    card = registry.get("resnet50")
    assert card.task == "classification"
    assert card.architecture == "resnet50"


def test_lookup_with_explicit_version(registry: ModelRegistry) -> None:
    card = registry.get("yolov8n", "1.2.0")
    assert card.key == "yolov8n@1.2.0"


def test_missing_model_raises(registry: ModelRegistry) -> None:
    with pytest.raises(ModelNotFoundError, match="no model named"):
        registry.get("no-such-model")


def test_missing_version_raises(registry: ModelRegistry) -> None:
    with pytest.raises(ModelNotFoundError, match="has no version"):
        registry.get("resnet50", "9.9.9")


def test_highest_version_wins(card_template: dict[str, Any]) -> None:
    older = ModelCard.model_validate({**card_template, "version": "1.9.0"})
    newer = ModelCard.model_validate({**card_template, "version": "1.10.0"})
    assert ModelRegistry([older, newer]).get("tiny-net").version == "1.10.0"


def test_duplicate_cards_rejected(card_template: dict[str, Any]) -> None:
    card = ModelCard.model_validate(card_template)
    with pytest.raises(DuplicateModelError):
        ModelRegistry([card, card])


def test_select_filters(registry: ModelRegistry) -> None:
    assert [c.name for c in registry.select(task="detection")] == ["yolov8n"]
    assert [c.name for c in registry.select(tag="imagenet")] == ["resnet50"]
    assert registry.select(task="segmentation") == []
    assert [c.name for c in registry.select(metric="top1_accuracy", min_value=0.7)] == ["resnet50"]
    assert registry.select(metric="top1_accuracy", min_value=0.99) == []


def test_names_are_sorted(registry: ModelRegistry) -> None:
    assert registry.names == sorted(registry.names)


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="does not exist"):
        ModelRegistry.from_directory(tmp_path / "nope")


def test_malformed_card_reports_its_path(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("name: broken\ntask: classification\n", encoding="utf-8")
    with pytest.raises(RegistryError, match="broken.yaml"):
        ModelRegistry.from_directory(tmp_path)

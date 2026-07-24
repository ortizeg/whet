"""The model-card schema must accept good cards and reject bad ones."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from ${package_name}.model_card import ModelCard


def test_shipped_cards_validate(card_paths: list[Path]) -> None:
    for path in card_paths:
        card = ModelCard.from_yaml(path)
        assert card.name
        assert len(card.weights.sha256) == 64


def test_template_card_is_valid(card_template: dict[str, Any]) -> None:
    card = ModelCard.model_validate(card_template)
    assert card.key == "tiny-net@0.1.0"
    assert card.inputs.shape == (3, 32, 32)
    assert card.status == "active"


def test_card_is_frozen(card_template: dict[str, Any]) -> None:
    card = ModelCard.model_validate(card_template)
    with pytest.raises(ValidationError):
        card.name = "other"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sha256", "not-a-digest"),
        ("sha256", "ABCD" * 16),  # uppercase is rejected: digests are lowercase hex
        ("sha256", "ab" * 31),  # too short
        ("url", "ftp://models.example.com/tiny-net.pth"),
    ],
)
def test_bad_weights_are_rejected(card_template: dict[str, Any], field: str, value: str) -> None:
    weights = dict(card_template["weights"])
    weights[field] = value
    payload = {**card_template, "weights": weights}
    with pytest.raises(ValidationError):
        ModelCard.model_validate(payload)


def test_unknown_field_is_rejected(card_template: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelCard.model_validate({**card_template, "trainig_dataset": "typo"})


def test_unknown_task_is_rejected(card_template: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ModelCard.model_validate({**card_template, "task": "telekinesis"})


def test_normalization_length_must_match_channels(card_template: dict[str, Any]) -> None:
    payload = {
        **card_template,
        "inputs": {"channels": 3, "height": 32, "width": 32, "mean": [0.5]},
    }
    with pytest.raises(ValidationError):
        ModelCard.model_validate(payload)


def test_metric_lookup(card_paths: list[Path]) -> None:
    cards = {card.name: card for card in (ModelCard.from_yaml(p) for p in card_paths)}
    resnet = cards["resnet50"]
    assert resnet.metric("top1_accuracy") == pytest.approx(0.7613)
    assert resnet.metric("does_not_exist") is None


def test_roundtrip_through_yaml(card_template: dict[str, Any]) -> None:
    card = ModelCard.model_validate(card_template)
    assert ModelCard.model_validate_json(card.model_dump_json()) == card
    assert "tiny-net" in card.to_yaml()
